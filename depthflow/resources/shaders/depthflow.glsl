/*
// (c) 2023-2025 CC BY-SA 4.0, Tremeschin.
// Refactored: Dual Layer with Standard DepthMake (Fixed Declaration Conflict)
*/

/* ---------------------------------------------------------------------------------------------- */

#ifndef DEPTHFLOW
#define DEPTHFLOW

struct DepthFlow {
    float quality;
    float height;
    float steady;
    float focus;
    float zoom;
    float isometric;
    float dolly;
    float invert;
    bool mirror;
    vec2 offset;
    vec2 center;
    vec2 origin;
    bool glued;
    // Output
    float derivative;
    float steep;
    float value;
    vec3 normal;
    vec2 gluv;
    bool oob;
};

// Standard DepthMake function to ensure correct interaction and coordinate system
DepthFlow DepthMake(
    Camera camera,
    DepthFlow depth,
    sampler2D depthmap
) {
    // Convert absolute values to relative values
    float rel_focus  = (depth.focus  * depth.height);
    float rel_steady = (depth.steady * depth.height);

    // Inject parallax options on the camera
    camera.position.xy += depth.offset;
    camera.isometric   += depth.isometric;
    camera.dolly       += depth.dolly;
    camera.zoom        += (depth.zoom - 1.0);
    camera.focal_length = (1.0 - rel_focus);
    camera.plane_point  = vec3(0.0, 0.0, 1.0);
    camera              = CameraProject(camera);
    depth.oob           = camera.out_of_bounds;

    // If out of bounds, return (but in dual layer logic, we might want background to continue, keeping for now)
    if (depth.oob)
        return depth;

    // Shift ray origin to the target center point
    camera.origin += vec3(depth.origin, 0);

    // Point where the ray intersects with a fixed point pivoting around depth=steady
    vec3 intersect = vec3(depth.center + camera.gluv, 1.0)
        - vec3(camera.position.xy, 0.0) * (1.0/(1.0 - rel_steady)) * int(depth.glued);

    // The quality of the parallax effect is how tiny the steps are
    float quality = (1.0 / mix(200, 2000, depth.quality));
    float probe   = (1.0 / mix( 50,  120, depth.quality));
    float safe = (1.0 - depth.height);
    float last_value = 0.0;
    float walk = 0.0;

    /* Main loop: Find the intersection with the scene */
    for (int stage=0; stage<2; stage++) {
        bool FORWARD  = (stage == 0);
        bool BACKWARD = (stage == 1);

        for (int it=0; it<1000; it++) {
            if (FORWARD && walk > 1.0)
                break;
            walk += (FORWARD ? probe : -quality);

            vec3 point = mix(camera.origin, intersect, mix(safe, 1.0, walk));
            depth.gluv = point.xy;

            last_value = depth.value;
            depth.value = gtexture(depthmap, depth.gluv, depth.mirror).r;

            float surface = depth.height * mix(depth.value, 1.0 - depth.value, depth.invert);
            float ceiling = (1.0 - point.z);

            if (ceiling < surface) {
                if (FORWARD) break;
            } else if (BACKWARD) {
                depth.derivative = (last_value - depth.value) / quality;
                break;
            }
        }
    }

    // The gradient is always normal to a surface;
    depth.normal = normalize(vec3(
        (gtexture(depthmap, depth.gluv - vec2(quality, 0), depth.mirror).r - depth.value) / quality,
        (gtexture(depthmap, depth.gluv - vec2(0, quality), depth.mirror).r - depth.value) / quality,
        max(depth.height, quality)
    ));

    depth.steep = depth.derivative * angle(depth.normal, vec3(0, 0, 1));
    return depth;
}

#define GetDepthFlow(name) \
    DepthFlow name; \
    { \
        name.isometric = name##Isometric; \
        name.dolly     = name##Dolly; \
        name.zoom      = name##Zoom; \
        name.offset    = name##Offset; \
        name.height    = name##Height; \
        name.focus     = name##Focus; \
        name.center    = name##Center; \
        name.steady    = name##Steady; \
        name.origin    = name##Origin; \
        name.mirror    = name##Mirror; \
        name.invert    = name##Invert; \
        name.quality   = iQuality; \
        name.glued     = true; \
        name.value     = 0.0; \
        name.gluv      = vec2(0.0); \
        name.oob       = false; \
        name.derivative = 1.0; \
    }
#endif

/* ---------------------------------------------------------------------------------------------- */

// ShaderFlow automatically injects uniforms: sampler2D image, depth, image_bg, depth_bg, subject_mask
// [Fixed] Removed manual uniform declarations to avoid C1038 error

void main() {
    GetCamera(iCamera);
    GetDepthFlow(iDepth);

    // === 1. Calculate Foreground (Foreground) ===
    // Use passed foreground depth map 'depth'
    DepthFlow fg = DepthMake(iCamera, iDepth, depth);

    // === 2. Calculate Background (Background) ===
    // Create background parameters, force mirror on to prevent black edges
    DepthFlow bg_params = iDepth;
    bg_params.mirror = true;

    // Background parallax logic
    DepthFlow bg = DepthMake(iCamera, bg_params, depth_bg);

    // === 3. Compositing ===

    // Base color: Foreground
    vec4 color_fg = gtexture(image, fg.gluv, iDepth.mirror);

    // Fill color: Background
    vec4 color_bg = gtexture(image_bg, bg.gluv, true);

    // === 4. Subject Mask Detection ===
    // Check if current screen position corresponds to a subject region in the original image
    // Use original UV coordinates (astuv) to sample the subject mask
    // This tells us if the current screen pixel should show subject content
    float subject_mask_value = gtexture(subject_mask, astuv, false).r;
    
    // Also check the sampled foreground position to see if it's in subject region
    // This helps when the view has shifted and we're sampling from a different part
    float fg_subject_mask = gtexture(subject_mask, fg.gluv, iDepth.mirror).r;
    
    // Use the maximum of both to be more conservative about showing background
    float combined_subject_mask = max(subject_mask_value, fg_subject_mask);
    
    // Calculate base mask for background blending:
    // When steep (steepness) is high, it means foreground is tearing, need to show background.
    // When fg.oob (out of bounds), also show background.
    float base_mask = smoothstep(iInpaintLimit, iInpaintLimit + 0.1, fg.steep);
    
    // === 5. Smart Masking: Prevent background from showing in subject regions ===
    // If we're in a subject region, we should prioritize showing the subject
    // from another angle rather than the background, even when there's some tearing.
    // Only show background if:
    //   1. We're NOT in a subject region, OR
    //   2. The foreground is severely torn (very high steep) even in subject region
    
    // Determine if we're in a subject region (threshold at 0.3 for more lenient detection)
    float subject_region = smoothstep(0.2, 0.5, combined_subject_mask);
    
    // In subject regions, require much higher steepness threshold to show background
    // This prevents background from showing when subject parts are just slightly occluded
    float subject_steep_threshold = iInpaintLimit * 3.0; // Triple the threshold in subject regions
    float subject_steep_mask = smoothstep(subject_steep_threshold, subject_steep_threshold + 0.3, fg.steep);
    
    // Combine masks: 
    // - Outside subject regions: use normal base_mask
    // - Inside subject regions: only show background if steepness is very high
    float final_mask = mix(
        base_mask,                          // Outside subject: use normal threshold
        subject_steep_mask,                 // Inside subject: use much stricter threshold
        subject_region
    );
    
    // Handle out-of-bounds: only show background if not in strong subject region
    // This prevents background from showing when we're just viewing subject from a different angle
    if (fg.oob) {
        // If we're in a subject region, try to keep showing subject even when slightly OOB
        // Only show background if we're clearly outside subject region
        if (combined_subject_mask < 0.5) {
            final_mask = 1.0;  // Show background only if not in subject region
        } else {
            // In subject region but OOB: reduce background visibility
            final_mask = mix(0.0, 1.0, smoothstep(0.5, 0.8, fg.steep));
        }
    }

    // Mix foreground and background
    fragColor = mix(color_fg, color_bg, final_mask);

    // === 4. Post Processing (Kept as is) ===

    if (iVigEnable) {
        vec2 away = astuv * (1.0 - astuv.yx);
        float linear = iVigDecay * (away.x*away.y);
        fragColor.rgb *= clamp(pow(linear, iVigIntensity), 0.0, 1.0);
    }

    if (iColorsSaturation != 1.0) {
        vec3 _hsv = rgb2hsv(fragColor.rgb);
        _hsv.y = clamp(_hsv.y * iColorsSaturation, 0.0, 1.0);
        fragColor.rgb = hsv2rgb(_hsv);
    }
    if (iColorsContrast != 1.0) {
        fragColor.rgb = clamp((fragColor.rgb - 0.5) * iColorsContrast + 0.5, 0.0, 1.0);
    }
    if (iColorsBrightness != 1.0) {
        fragColor.rgb = clamp(fragColor.rgb * iColorsBrightness, 0.0, 1.0);
    }
    if (iColorsGamma != 1.0) {
        fragColor.rgb = pow(fragColor.rgb, vec3(1.0/iColorsGamma));
    }
    if (iColorsSepia != 0.0) {
        float luminance = dot(fragColor.rgb, vec3(0.299, 0.587, 0.114));
        fragColor.rgb = mix(fragColor.rgb, luminance*vec3(1.2, 1.0, 0.8), iColorsSepia);
    }
    if (iColorsGrayscale != 0.0) {
        float luminance = dot(fragColor.rgb, vec3(0.299, 0.587, 0.114));
        fragColor.rgb = mix(fragColor.rgb, vec3(luminance), iColorsGrayscale);
    }
}