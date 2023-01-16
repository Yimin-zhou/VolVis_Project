#include "gradient_volume.h"
#include <algorithm>
#include <exception>
#include <glm/geometric.hpp>
#include <glm/vector_relational.hpp>
#include <gsl/span>

namespace volume {

// Compute the maximum magnitude from all gradient voxels
static float computeMaxMagnitude(gsl::span<const GradientVoxel> data)
{
    return std::max_element(
        std::begin(data),
        std::end(data),
        [](const GradientVoxel& lhs, const GradientVoxel& rhs) {
            return lhs.magnitude < rhs.magnitude;
        })
        ->magnitude;
}

// Compute the minimum magnitude from all gradient voxels
static float computeMinMagnitude(gsl::span<const GradientVoxel> data)
{
    return std::min_element(
        std::begin(data),
        std::end(data),
        [](const GradientVoxel& lhs, const GradientVoxel& rhs) {
            return lhs.magnitude < rhs.magnitude;
        })
        ->magnitude;
}

// Compute a gradient volume from a volume
static std::vector<GradientVoxel> computeGradientVolume(const Volume& volume)
{
    const auto dim = volume.dims();

    std::vector<GradientVoxel> out(static_cast<size_t>(dim.x * dim.y * dim.z));
    for (int z = 1; z < dim.z - 1; z++) {
        for (int y = 1; y < dim.y - 1; y++) {
            for (int x = 1; x < dim.x - 1; x++) {
                const float gx = (volume.getVoxel(x + 1, y, z) - volume.getVoxel(x - 1, y, z)) / 2.0f;
                const float gy = (volume.getVoxel(x, y + 1, z) - volume.getVoxel(x, y - 1, z)) / 2.0f;
                const float gz = (volume.getVoxel(x, y, z + 1) - volume.getVoxel(x, y, z - 1)) / 2.0f;

                const glm::vec3 v { gx, gy, gz };
                const size_t index = static_cast<size_t>(x + dim.x * (y + dim.y * z));
                out[index] = GradientVoxel { v, glm::length(v) };
            }
        }
    }
    return out;
}

GradientVolume::GradientVolume(const Volume& volume)
    : m_dim(volume.dims())
    , m_data(computeGradientVolume(volume))
    , m_minMagnitude(computeMinMagnitude(m_data))
    , m_maxMagnitude(computeMaxMagnitude(m_data))
{
}

float GradientVolume::maxMagnitude() const
{
    return m_maxMagnitude;
}

float GradientVolume::minMagnitude() const
{
    return m_minMagnitude;
}

glm::ivec3 GradientVolume::dims() const
{
    return m_dim;
}

// check if the point is in range
bool checkBoundaryX(const glm::vec3& dim, const glm::vec3& coord)
{
    if (glm::any(glm::lessThan(coord, glm::vec3(0))) || glm::any(glm::greaterThanEqual(coord, glm::vec3(dim)))) {
        return false;
    } else {
        return true;
    }
}

glm::vec3 addVec(const glm::vec3& vec1,const glm::vec3& vec2) {
    float x = vec1.x + vec2.x;
    float y = vec1.y + vec2.y;
    float z = vec1.z + vec2.z;

    return glm::vec3(x, y, z);
}

glm::vec3 subVec(const glm::vec3& vec1, const glm::vec3& vec2)
{
    float x = vec1.x - vec2.x;
    float y = vec1.y - vec2.y;
    float z = vec1.z - vec2.z;

    return glm::vec3(x, y, z);
}





// This function returns a gradientVoxel at coord based on the current interpolation mode.
GradientVoxel GradientVolume::getGradientInterpolate(const glm::vec3& coord) const
{
    switch (interpolationMode) {
    case InterpolationMode::NearestNeighbour: {
        return getGradientNearestNeighbor(coord);
    }
    case InterpolationMode::Linear: {
        return getGradientLinearInterpolate(coord);
    }
    case InterpolationMode::Cubic: {
        // No cubic in this case, linear is good enough for the gradient.
        return getGradientLinearInterpolate(coord);
    }
    default: {
        throw std::exception();
    }
    };
}

// This function returns the nearest neighbour given a position in the volume given by coord.
// Notice that in this framework we assume that the distance between neighbouring voxels is 1 in all directions
GradientVoxel GradientVolume::getGradientNearestNeighbor(const glm::vec3& coord) const
{
    if (glm::any(glm::lessThan(coord, glm::vec3(0))) || glm::any(glm::greaterThanEqual(coord, glm::vec3(m_dim))))
        return { glm::vec3(0.0f), 0.0f };

    auto roundToPositiveInt = [](float f) {
        return static_cast<int>(f + 0.5f);
    };

    return getGradient(roundToPositiveInt(coord.x), roundToPositiveInt(coord.y), roundToPositiveInt(coord.z));
}

// ======= TODO : IMPLEMENT ========
// Returns the trilinearly interpolated gradinet at the given coordinate.
// Use the linearInterpolate function that you implemented below.
GradientVoxel GradientVolume::getGradientLinearInterpolate(const glm::vec3& coord) const
{
    // TODO check boundary
    if (glm::any(glm::lessThan(coord, glm::vec3(0))) || glm::any(glm::greaterThanEqual(coord, glm::vec3(m_dim))))
        return { glm::vec3(0.0f), 0.0f };

    // Apply bilinear interploations to x-y coordinates
    GradientVoxel b0 = biLinearInterpolation(glm::vec2(coord.x, coord.y), floor(coord.z));
    GradientVoxel b1 = biLinearInterpolation(glm::vec2(coord.x, coord.y), ceil(coord.z));

    glm::ivec3 topLeftCoordBack = glm::ivec3(floor(coord.x), ceil(coord.y), floor(coord.z));
    glm::ivec3 topRightCoordBack = glm::ivec3(ceil(coord.x), ceil(coord.y), floor(coord.z));
    glm::ivec3 bottomLeftCoordFront = glm::ivec3(floor(coord.x), floor(coord.y), ceil(coord.z));

    // Apply final interpolation to the third direction to get final value
    float c = (coord.z - (float)topLeftCoordBack.z) / ((float)bottomLeftCoordFront.z - (float)topRightCoordBack.z);
    return linearInterpolate(b0, b1, c);
}

GradientVoxel GradientVolume::biLinearInterpolation(const glm::vec2& xyCoord, int z) const
{

    // Getting the nearest neighbors in the x-y directions (2 in x and 2 in y direction)
    glm::ivec2 topLeftCoord = glm::ivec2(floor(xyCoord.x), ceil(xyCoord.y)); // top-left
    glm::ivec2 bottomLeftCoord = glm::ivec2(floor(xyCoord.x), floor(xyCoord.y)); // bottom-left
    glm::ivec2 topRightCoord = glm::ivec2(ceil(xyCoord.x), ceil(xyCoord.y)); // top-right
    glm::ivec2 bottomRightCoord = glm::ivec2(ceil(xyCoord.x), floor(xyCoord.y)); // bottom-right 

    GradientVoxel voxelTopLeft = { glm::vec3(0.0f), 0.0f };
    GradientVoxel voxelBottomLeft = { glm::vec3(0.0f), 0.0f };
    GradientVoxel voxelTopRight = { glm::vec3(0.0f), 0.0f };
    GradientVoxel voxelBottomRight = { glm::vec3(0.0f), 0.0f };


    //Ensuring boundaries present
    if (checkBoundaryX(glm::vec3(m_dim), glm::vec3(topLeftCoord.x, topLeftCoord.y, z)))
        voxelTopLeft = getGradient(floor(xyCoord.x), ceil(xyCoord.y), z);
    if (checkBoundaryX(glm::vec3(m_dim), glm::vec3(bottomLeftCoord.x, bottomLeftCoord.y, z)))
        voxelBottomLeft = getGradient(bottomLeftCoord.x, bottomLeftCoord.y, z);
    if (checkBoundaryX(glm::vec3(m_dim), glm::vec3(topRightCoord.x, topRightCoord.y, z)))
        voxelTopRight = getGradient(topRightCoord.x, topRightCoord.y, z);
    if (checkBoundaryX(glm::vec3(m_dim), glm::vec3(bottomRightCoord.x, bottomRightCoord.y, z)))
        voxelBottomRight = getGradient(bottomRightCoord.x, bottomRightCoord.y, z);

    float a = 0.0f;
    float b = 0.0f;

    a = (xyCoord.x - (float)bottomLeftCoord.x) / ((float)topRightCoord.x - (float)topLeftCoord.x);
    b = ((float)bottomLeftCoord.y - xyCoord.y) / ((float)bottomLeftCoord.y - (float)topLeftCoord.y);

    GradientVoxel linear1 = linearInterpolate(voxelTopLeft, voxelTopRight, a);
    GradientVoxel linear2 = linearInterpolate(voxelBottomLeft, voxelBottomRight, a);

    return linearInterpolate(linear1, linear2, b);
}

// ======= TODO : IMPLEMENT ========
// This function should linearly interpolates the value from g0 to g1 given the factor (t).
// At t=0, linearInterpolate should return g0 and at t=1 it returns g1.
GradientVoxel GradientVolume::linearInterpolate(const GradientVoxel& g0, const GradientVoxel& g1, float factor)
{
    //(g1.dir - g0.dir) * factor + g0.dir;
    glm::vec3 result = addVec(subVec(g1.dir, g0.dir) * factor, g0.dir);        
    return GradientVoxel { result, glm::length(result)};
}

// This function returns a gradientVoxel without using interpolation
GradientVoxel GradientVolume::getGradient(int x, int y, int z) const
{
    const size_t i = static_cast<size_t>(x + m_dim.x * (y + m_dim.y * z));
    return m_data[i];
}



}



