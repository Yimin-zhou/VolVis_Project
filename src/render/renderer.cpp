#include "renderer.h"
#include <algorithm>
#include <algorithm> // std::fill
#include <cmath>
#include <functional>
#include <glm/common.hpp>
#include <glm/gtx/component_wise.hpp>
#include <iostream>
#include <tbb/blocked_range2d.h>
#include <tbb/parallel_for.h>
#include <tuple>
#include <string>

namespace render {

// The renderer is passed a pointer to the volume, gradinet volume, camera and an initial renderConfig.
// The camera being pointed to may change each frame (when the user interacts). When the renderConfig
// changes the setConfig function is called with the updated render config. This gives the Renderer an
// opportunity to resize the framebuffer.
Renderer::Renderer(
    const volume::Volume* pVolume,
    const volume::GradientVolume* pGradientVolume,
    const render::RayTraceCamera* pCamera,
    const RenderConfig& initialConfig)
    : m_pVolume(pVolume)
    , m_pGradientVolume(pGradientVolume)
    , m_pCamera(pCamera)
    , m_config(initialConfig)
{
    resizeImage(initialConfig.renderResolution);
}

// Set a new render config if the user changed the settings.
void Renderer::setConfig(const RenderConfig& config)
{
    if (config.renderResolution != m_config.renderResolution)
        resizeImage(config.renderResolution);

    m_config = config;
}

// Resize the framebuffer and fill it with black pixels.
void Renderer::resizeImage(const glm::ivec2& resolution)
{
    m_frameBuffer.resize(size_t(resolution.x) * size_t(resolution.y), glm::vec4(0.0f));
}

// Clear the framebuffer by setting all pixels to black.
void Renderer::resetImage()
{
    std::fill(std::begin(m_frameBuffer), std::end(m_frameBuffer), glm::vec4(0.0f));
}

// Return a VIEW into the framebuffer. This view is merely a reference to the m_frameBuffer member variable.
// This does NOT make a copy of the framebuffer.
gsl::span<const glm::vec4> Renderer::frameBuffer() const
{
    return m_frameBuffer;
}

// Main render function. It computes an image according to the current renderMode.
// Multithreading is enabled in Release/RelWithDebInfo modes. In Debug mode multithreading is disabled to make debugging easier.
void Renderer::render()
{
    resetImage();

    static constexpr float sampleStep = 1.0f;
    const glm::vec3 planeNormal = -glm::normalize(m_pCamera->forward());
    const glm::vec3 volumeCenter = glm::vec3(m_pVolume->dims()) / 2.0f;
    const Bounds bounds { glm::vec3(0.0f), glm::vec3(m_pVolume->dims() - glm::ivec3(1)) };

    // 0 = sequential (single-core), 1 = TBB (multi-core)
#ifdef NDEBUG
    // If NOT in debug mode then enable parallelism using the TBB library (Intel Threaded Building Blocks).
#define PARALLELISM 1
#else
    // Disable multi threading in debug mode.
#define PARALLELISM 0
#endif

#if PARALLELISM == 0
    // Regular (single threaded) for loops.
    for (int x = 0; x < m_config.renderResolution.x; x++) {
        for (int y = 0; y < m_config.renderResolution.y; y++) {
#else
    // Parallel for loop (in 2 dimensions) that subdivides the screen into tiles.
    const tbb::blocked_range2d<int> screenRange { 0, m_config.renderResolution.y, 0, m_config.renderResolution.x };
        tbb::parallel_for(screenRange, [&](tbb::blocked_range2d<int> localRange) {
        // Loop over the pixels in a tile. This function is called on multiple threads at the same time.
        for (int y = std::begin(localRange.rows()); y != std::end(localRange.rows()); y++) {
            for (int x = std::begin(localRange.cols()); x != std::end(localRange.cols()); x++) {
#endif
            // Compute a ray for the current pixel.
            const glm::vec2 pixelPos = glm::vec2(x, y) / glm::vec2(m_config.renderResolution);
            Ray ray = m_pCamera->generateRay(pixelPos * 2.0f - 1.0f);

            // Compute where the ray enters and exists the volume.
            // If the ray misses the volume then we continue to the next pixel.
            if (!instersectRayVolumeBounds(ray, bounds))
                continue;

            // Get a color for the current pixel according to the current render mode.
            glm::vec4 color {};
            switch (m_config.renderMode) {
            case RenderMode::RenderSlicer: {
                color = traceRaySlice(ray, volumeCenter, planeNormal);
                break;
            }
            case RenderMode::RenderMIP: {
                color = traceRayMIP(ray, sampleStep);
                break;
            }
            case RenderMode::RenderComposite: {
                color = traceRayComposite(ray, sampleStep);
                break;
            }
            case RenderMode::RenderIso: {
                color = traceRayISO(ray, sampleStep);
                break;
            }
            case RenderMode::RenderTF2D: {
                color = traceRayTF2D(ray, sampleStep);
                break;
            }
            };
            // Write the resulting color to the screen.
            fillColor(x, y, color);

#if PARALLELISM == 1
        }
    }
});
#else
            }
        }
#endif
}

//glm::vec3 addVector(const glm::vec3& vec1, const glm::vec3& vec2)
//{
//    float x = vec1.x + vec2.x;
//    float y = vec1.y + vec2.y;
//    float z = vec1.z + vec2.z;
//
//
//    glm::vec3 v1(1.f, 1.f, 1.f);
//    glm::vec3 v2(2.f, 2.f, 2.f);
//    glm::vec3 v3 = v1 + v2;
//
//    return glm::vec3(x, y, z);
//}

// ======= DO NOT MODIFY THIS FUNCTION ========
// This function generates a view alongside a plane perpendicular to the camera through the center of the volume
//  using the slicing technique.
glm::vec4 Renderer::traceRaySlice(const Ray& ray, const glm::vec3& volumeCenter, const glm::vec3& planeNormal) const
{
    const float t = glm::dot(volumeCenter - ray.origin, planeNormal) / glm::dot(ray.direction, planeNormal);
    const glm::vec3 samplePos = ray.origin + ray.direction * t;
    const float val = m_pVolume->getSampleInterpolate(samplePos);
    return glm::vec4(glm::vec3(std::max(val / m_pVolume->maximum(), 0.0f)), 1.f);
}

// ======= DO NOT MODIFY THIS FUNCTION ========
// Function that implements maximum-intensity-projection (MIP) raycasting.
// It returns the color assigned to a ray/pixel given it's origin, direction and the distances
// at which it enters/exits the volume (ray.tmin & ray.tmax respectively).
// The ray must be sampled with a distance defined by the sampleStep
glm::vec4 Renderer::traceRayMIP(const Ray& ray, float sampleStep) const
{
    float maxVal = 0.0f;

    // Incrementing samplePos directly instead of recomputing it each frame gives a measureable speed-up.
    glm::vec3 samplePos = ray.origin + ray.tmin * ray.direction;
    const glm::vec3 increment = sampleStep * ray.direction;
    for (float t = ray.tmin; t <= ray.tmax; t += sampleStep, samplePos += increment) {
        const float val = m_pVolume->getSampleInterpolate(samplePos);
        maxVal = std::max(val, maxVal);
    }
    // Normalize the result to a range of [0 to mpVolume->maximum()].
    return glm::vec4(glm::vec3(maxVal) / m_pVolume->maximum(), 1.0f);
}

// ======= TODO: IMPLEMENT ========
// This function should find the position where the ray intersects with the volume's isosurface.
// If volume shading is DISABLED then simply return the isoColor.
// If volume shading is ENABLED then return the phong-shaded color at that location using the local gradient (from m_pGradientVolume).
//   Use the camera position (m_pCamera->position()) as the light position.
// Use the bisectionAccuracy function (to be implemented) to get a more precise isosurface location between two steps.
glm::vec4 Renderer::traceRayISO(const Ray& ray, float sampleStep) const
{
    static constexpr glm::vec3 isoColor { 0.8f, 0.8f, 0.2f };

    // Incrementing samplePos directly instead of recomputing it each frame gives a measureable speed-up.
    glm::vec3 samplePos = ray.origin + ray.tmin * ray.direction;
    glm::vec3 preSamplePos = samplePos;
    glm::vec3 accuratePos = glm::vec3(0);
    float preT = 0.0f;
    bool isHit = false;
    const glm::vec3 increment = sampleStep * ray.direction;
    for (float t = ray.tmin; t <= ray.tmax; t += sampleStep, samplePos += increment) 
    {
        const float val = m_pVolume->getSampleInterpolate(samplePos);
        if (m_config.isoValue < val) 
        {
             accuratePos = ray.origin + ray.tmin * ray.direction
                + (bisectionAccuracy(ray, preT, t, m_config.isoValue) - ray.tmin) * ray.direction;
            isHit = true;
            break;
        }
        preSamplePos = samplePos;
        preT = t;
    }

    if (isHit)
    {
        glm::vec3 result_color = isoColor;
        if (m_config.volumeShading) {
            //We use the camera position for the light vector, however reverse it as we need the light going from the object to camera and not nice versa
            glm::vec3 L = -1.0f * glm::normalize(m_pCamera->position());
            result_color = computePhongShading(isoColor, m_pGradientVolume->getGradientInterpolate(accuratePos), L, m_pCamera->position());
        }
            
        return glm::vec4(result_color, 1.0f);
    }
    else
    {
        return glm::vec4(glm::vec3(0), 0.0f);
    }
}

// ======= TODO: IMPLEMENT ========
// Given that the iso value lies somewhere between t0 and t1, find a t for which the value
// closely matches the iso value (less than 0.01 difference). Add a limit to the number of
// iterations such that it does not get stuck in degerate cases.
float Renderer::bisectionAccuracy(const Ray& ray, float t0, float t1, float isoValue) const
{
    glm::vec3 startPos = ray.origin + ray.tmin * ray.direction;

    float l = t0;
    float r = t1;
    float m = 0;
    int iteration = 50;

    while (l <= r || iteration < 0) {
        m = (r - l) / 2.0f + l;
        glm::vec3 middilPos = startPos + (m - ray.tmin) * ray.direction;
        if (m_pVolume->getSampleInterpolate(middilPos) < isoValue - 0.01f) {
            l = m + 0.01;
        } else if (m_pVolume->getSampleInterpolate(middilPos) > isoValue + 0.01f) {
            r = m - 0.01;
        } else {
            return m;
        }
        --iteration;
    }
    return m;
}

// ======= TODO: IMPLEMENT ========
// Compute Phong Shading given the voxel color (material color), the gradient, the light vector and view vector.
// You can find out more about the Phong shading model at:
// https://en.wikipedia.org/wiki/Phong_reflection_model
//
// Use the given color for the ambient/specular/diffuse (you are allowed to scale these constants by a scalar value).
// You are free to choose any specular power that you'd like.
glm::vec3 Renderer::computePhongShading(const glm::vec3& color, const volume::GradientVoxel& gradient, const glm::vec3& L, const glm::vec3& V)
{
    const float k_a = 0.1;
    const float k_d = 0.7;
    const float k_s = 0.2;
    const float alpha = 100;
    glm::vec3 zero_vec = glm::vec3(0.0f);

    // Get normalized vectors
    glm::vec3 gradient_hat = gradient.magnitude != 0 ? glm::normalize(gradient.dir) : zero_vec;
    glm::vec3 L_hat = glm::length(L) != 0 ? (L) : zero_vec;
    glm::vec3 V_hat = glm::length(V) != 0 ? glm::normalize(V) : zero_vec;
   
    // Get reflection vector
    glm::vec3 R = 2.0f * glm::dot(L_hat ,gradient_hat) * gradient_hat - L_hat;
    glm::vec3 R_hat = glm::length(R) != 0 ? glm::normalize(R) : zero_vec;

    glm::vec3 specular = k_s * pow(glm::dot(R_hat, V_hat), alpha) * color;
    glm::vec3 ambient = k_a * color;
    glm::vec3 diffuse = k_d * glm::dot(L_hat, gradient_hat) * color;

    //Calculate illumination
    glm::vec3 i_p = ambient + specular + diffuse;

    return i_p;
}

// ======= TODO: IMPLEMENT ========
// In this function, implement 1D transfer function raycasting.
// Use getTFValue to compute the color for a given volume value according to the 1D transfer function.
glm::vec4 Renderer::traceRayComposite(const Ray& ray, float sampleStep) const
{
    glm::vec3 samplePos = ray.origin + ray.tmin * ray.direction;
    const glm::vec3 increment = sampleStep * ray.direction;

    glm::vec4 sampleColor, preSampleColor  = glm::vec4(0.0f);

    for (float t = ray.tmin; t <= ray.tmax; t += sampleStep, samplePos += increment) 
    {
        const float val = m_pVolume->getSampleInterpolate(samplePos);

        // front to back
        glm::vec4 c = getTFValue(val);
        // phong shading
        glm::vec3 L = -2.0f * glm::normalize(m_pCamera->position());
        glm::vec3 shading = computePhongShading(glm::vec3(c), m_pGradientVolume->getGradientInterpolate(samplePos),
            L, m_pCamera->position());
        if (!glm::any(glm::isnan(shading)))
        {
            c = glm::vec4(shading, c.a);
        }

        sampleColor = preSampleColor + (1.0f - preSampleColor.a) * (glm::vec4(c.r * c.a, c.g * c.a, c.b * c.a, c.a));
        preSampleColor = sampleColor;
    }
    return sampleColor;
}

// ======= DO NOT MODIFY THIS FUNCTION ========
// Looks up the color+opacity corresponding to the given volume value from the 1D tranfer function LUT (m_config.tfColorMap).
// The value will initially range from (m_config.tfColorMapIndexStart) to (m_config.tfColorMapIndexStart + m_config.tfColorMapIndexRange) .
glm::vec4 Renderer::getTFValue(float val) const
{
    // Map value from [m_config.tfColorMapIndexStart, m_config.tfColorMapIndexStart + m_config.tfColorMapIndexRange) to [0, 1) .
    const float range01 = (val - m_config.tfColorMapIndexStart) / m_config.tfColorMapIndexRange;
    const size_t i = std::min(static_cast<size_t>(range01 * static_cast<float>(m_config.tfColorMap.size())), m_config.tfColorMap.size() - 1);
    return m_config.tfColorMap[i];
}

// ======= TODO: IMPLEMENT ========
// In this function, implement 2D transfer function raycasting.
// Use the getTF2DOpacity function that you implemented to compute the opacity according to the 2D transfer function.
glm::vec4 Renderer::traceRayTF2D(const Ray& ray, float sampleStep) const
{
    glm::vec3 samplePos = ray.origin + ray.tmin * ray.direction;
    const glm::vec3 increment = sampleStep * ray.direction;

    glm::vec4 sampleColor, preSampleColor = glm::vec4(0.0f);

    for (float t = ray.tmin; t <= ray.tmax; t += sampleStep, samplePos += increment) {
        const float val = m_pVolume->getSampleInterpolate(samplePos);
        const float gradientM = m_pGradientVolume->getGradientInterpolate(samplePos).magnitude;

        // front to back
        glm::vec4 c = m_config.TF2DColor;
        c.a = getTF2DOpacity(val, gradientM);
        sampleColor = preSampleColor + (1.0f - preSampleColor.a) * (glm::vec4(c.r * c.a, c.g * c.a, c.b * c.a, c.a));
        preSampleColor = sampleColor;
    }
    return sampleColor;
}

float cross(glm::vec2 p1, glm::vec2 p2, glm::vec2 p3)
{
    return (p2.x - p1.x) * (p3.y - p1.y) - (p2.y - p1.y) * (p3.x - p1.x);
}

bool isInTriangle(float triangleHeight, float triangleRadius, float traingleApex, glm::vec2 sample)
{
    glm::vec2 apex = glm::vec2(traingleApex, 0.0f);
    glm::vec2 right = glm::vec2(traingleApex + triangleRadius / 2.0f, triangleHeight);
    glm::vec2 left = glm::vec2(traingleApex - triangleRadius / 2.0f, triangleHeight);

    if (cross(apex, right, sample) > 0 && cross(right, left, sample) > 0 && cross(left, apex, sample) > 0)
        return true;
    return false;
}
// ======= TODO: IMPLEMENT ========
// This function should return an opacity value for the given intensity and gradient according to the 2D transfer function.
// Calculate whether the values are within the radius/intensity triangle defined in the 2D transfer function widget.
// If so: return a tent weighting as described in the assignment
// Otherwise: return 0.0f
//
// The 2D transfer function settings can be accessed through m_config.TF2DIntensity and m_config.TF2DRadius.
float Renderer::getTF2DOpacity(float intensity, float gradientMagnitude) const
{

    float triangleHeight = m_pGradientVolume->maxMagnitude() - m_pGradientVolume->minMagnitude();    

    //if (isInRangeOfTriangle(triangleHeight, m_config.TF2DRadius, m_config.TF2DIntensity, gradientMagnitude ,intensity)) {
    //    float x_traingle = (m_config.TF2DRadius * gradientMagnitude) / (2 * triangleHeight);

    //    // Calculate the ratio of distance of intensity from apex/ length from apex vertical to diagonal
    //    float ratio = (std::abs(m_config.TF2DIntensity - intensity) / x_traingle) * 1;

    //    // since value drops from 1 to 0 and not 0 to 1
    //    return (1 - ratio);
    //}

     if (isInTriangle(triangleHeight, m_config.TF2DRadius * 2, m_config.TF2DIntensity, glm::vec2(intensity, gradientMagnitude))) 
     {
        // m_config.TF2DRadius  is half of the triangle base line
        // Distance from center to left or right diagonal line (y = gradientMagnitude)
        float distance = (m_config.TF2DRadius) * (gradientMagnitude / triangleHeight);
        // Calculate the ratio of distance of intensity from center vertical line to diagonal line
        float ratio = abs(intensity - m_config.TF2DIntensity) / distance;
        return (1.0f - ratio) * m_config.TF2DColor.a;
     }

    return 0.0f;

}


// This function computes if a ray intersects with the axis-aligned bounding box around the volume.
// If the ray intersects then tmin/tmax are set to the distance at which the ray hits/exists the
// volume and true is returned. If the ray misses the volume the the function returns false.
//
// If you are interested you can learn about it at.
// https://www.scratchapixel.com/lessons/3d-basic-rendering/minimal-ray-tracer-rendering-simple-shapes/ray-box-intersection
bool Renderer::instersectRayVolumeBounds(Ray& ray, const Bounds& bounds) const
{
    const glm::vec3 invDir = 1.0f / ray.direction;
    const glm::bvec3 sign = glm::lessThan(invDir, glm::vec3(0.0f));

    float tmin = (bounds.lowerUpper[sign[0]].x - ray.origin.x) * invDir.x;
    float tmax = (bounds.lowerUpper[!sign[0]].x - ray.origin.x) * invDir.x;
    const float tymin = (bounds.lowerUpper[sign[1]].y - ray.origin.y) * invDir.y;
    const float tymax = (bounds.lowerUpper[!sign[1]].y - ray.origin.y) * invDir.y;

    if ((tmin > tymax) || (tymin > tmax))
        return false;
    tmin = std::max(tmin, tymin);
    tmax = std::min(tmax, tymax);

    const float tzmin = (bounds.lowerUpper[sign[2]].z - ray.origin.z) * invDir.z;
    const float tzmax = (bounds.lowerUpper[!sign[2]].z - ray.origin.z) * invDir.z;

    if ((tmin > tzmax) || (tzmin > tmax))
        return false;

    ray.tmin = std::max(tmin, tzmin);
    ray.tmax = std::min(tmax, tzmax);
    return true;
}

// This function inserts a color into the framebuffer at position x,y
void Renderer::fillColor(int x, int y, const glm::vec4& color)
{
    const size_t index = static_cast<size_t>(m_config.renderResolution.x * y + x);
    m_frameBuffer[index] = color;
}


// Determine the range of values of intensity for gradient to be inside the traingle
//  Given the triangle in the viewer is symmetric and is centred around m_config.TF2DIntensity
//  We use the fatch that tan(theta)==(m_config.TF2DRadius/2)/(gradient_max-gradient_m)==x_traingle/y
//  hence for given gradient, the point will lie inside the triangle only if x is within +/- x_traingle pf TF2DIntensity
bool Renderer::isInRangeOfTriangle(float traingleHeight, float triangleRadius, float traingleApex, float yCoord, float xCoord) const
{
    // get range of x of the traingle given y
    float x_traingle = (triangleRadius * yCoord) / (2 * traingleHeight);

    // ensuring limits
    float x_max = std::min(traingleApex + x_traingle, m_pVolume->maximum());
    float x_min = std::max(traingleApex - x_traingle, m_pVolume->minimum());

    // check if xCoord withing range of x_triangle else point not inside
    return (xCoord <= x_max && xCoord >= x_min) ? true : false;
}

}