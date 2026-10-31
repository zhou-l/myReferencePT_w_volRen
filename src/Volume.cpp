#include "Volume.h"

#include <algorithm>
#include <cmath>
#include <cstdio>

namespace Volume
{

bool LoadVol(const std::string& filename, DensityGrid& grid)
{
	FILE* file = fopen(filename.c_str(), "rb");
	if (!file) return false;

	// Header: "VOL", version byte, encoding, resolution, channel count, bounding box
	char magic[3];
	uint8_t version = 0;
	int32_t encoding = 0, xres = 0, yres = 0, zres = 0, channels = 0;
	float bbox[6];

	bool ok = fread(magic, 1, 3, file) == 3 && magic[0] == 'V' && magic[1] == 'O' && magic[2] == 'L'
		&& fread(&version, 1, 1, file) == 1 && version == 3
		&& fread(&encoding, 4, 1, file) == 1 && encoding == 1	//< 1 = float32
		&& fread(&xres, 4, 1, file) == 1 && fread(&yres, 4, 1, file) == 1 && fread(&zres, 4, 1, file) == 1
		&& fread(&channels, 4, 1, file) == 1 && fread(bbox, 4, 6, file) == 6
		&& xres > 0 && yres > 0 && zres > 0 && channels > 0;

	if (ok) {
		size_t voxelCount = size_t(xres) * size_t(yres) * size_t(zres);
		std::vector<float> data(voxelCount * size_t(channels));
		ok = fread(data.data(), sizeof(float), data.size(), file) == data.size();

		if (ok) {
			grid.nx = uint32_t(xres);
			grid.ny = uint32_t(yres);
			grid.nz = uint32_t(zres);
			grid.density.resize(voxelCount);
			for (size_t i = 0; i < voxelCount; i++) grid.density[i] = (std::max)(0.0f, data[i * channels]);
		}
	}

	fclose(file);
	return ok;
}

// Value noise helpers for the procedural cloud
static float hash3(int x, int y, int z)
{
	uint32_t h = uint32_t(x) * 73856093u ^ uint32_t(y) * 19349663u ^ uint32_t(z) * 83492791u;
	h ^= h >> 13; h *= 0x5bd1e995u; h ^= h >> 15;
	return float(h & 0xFFFFFF) / float(0xFFFFFF);
}

static float valueNoise(float x, float y, float z)
{
	int ix = int(std::floor(x)), iy = int(std::floor(y)), iz = int(std::floor(z));
	float fx = x - ix, fy = y - iy, fz = z - iz;

	// Smoothstep fade
	fx = fx * fx * (3.0f - 2.0f * fx);
	fy = fy * fy * (3.0f - 2.0f * fy);
	fz = fz * fz * (3.0f - 2.0f * fz);

	auto lerp = [](float a, float b, float t) { return a + (b - a) * t; };
	float x00 = lerp(hash3(ix, iy, iz), hash3(ix + 1, iy, iz), fx);
	float x10 = lerp(hash3(ix, iy + 1, iz), hash3(ix + 1, iy + 1, iz), fx);
	float x01 = lerp(hash3(ix, iy, iz + 1), hash3(ix + 1, iy, iz + 1), fx);
	float x11 = lerp(hash3(ix, iy + 1, iz + 1), hash3(ix + 1, iy + 1, iz + 1), fx);
	return lerp(lerp(x00, x10, fy), lerp(x01, x11, fy), fz);
}

static float fbm(float x, float y, float z)
{
	float sum = 0.0f, amplitude = 0.5f, total = 0.0f;
	for (int octave = 0; octave < 5; octave++) {
		sum += amplitude * valueNoise(x, y, z);
		total += amplitude;
		x *= 2.0f; y *= 2.0f; z *= 2.0f;
		amplitude *= 0.5f;
	}
	return sum / total;
}

void GenerateCloud(DensityGrid& grid, uint32_t resolution)
{
	grid.nx = grid.ny = grid.nz = resolution;
	grid.density.resize(size_t(resolution) * resolution * resolution);

	for (uint32_t z = 0; z < resolution; z++)
	for (uint32_t y = 0; y < resolution; y++)
	for (uint32_t x = 0; x < resolution; x++) {
		// Voxel center in [-1, 1]^3
		float px = (x + 0.5f) / resolution * 2.0f - 1.0f;
		float py = (y + 0.5f) / resolution * 2.0f - 1.0f;
		float pz = (z + 0.5f) / resolution * 2.0f - 1.0f;
		float r = std::sqrt(px * px + py * py + pz * pz);

		float n = fbm(px * 3.0f + 10.0f, py * 3.0f + 10.0f, pz * 3.0f + 10.0f);
		float d = (1.0f - r) * 2.0f + (n - 0.5f) * 2.5f - 0.6f;
		grid.density[(size_t(z) * resolution + y) * resolution + x] = (std::min)(1.0f, (std::max)(0.0f, d));
	}
}

void BuildMajorants(DensityGrid& grid, uint32_t resolution)
{
	grid.mx = grid.my = grid.mz = resolution;
	grid.majorants.assign(size_t(resolution) * resolution * resolution, 0.0f);

	// Range of voxels whose values can influence trilinear lookups in [lo, hi] along one axis
	// (voxel centers sit at (i + 0.5) / n, lookups clamp to the edge)
	auto voxelRange = [](uint32_t cell, uint32_t cells, uint32_t n, int& first, int& last) {
		float lo = float(cell) / cells * n - 0.5f;
		float hi = float(cell + 1) / cells * n - 0.5f;
		first = (std::max)(0, int(std::floor(lo)));
		last = (std::min)(int(n) - 1, int(std::ceil(hi)));
	};

	for (uint32_t cz = 0; cz < resolution; cz++)
	for (uint32_t cy = 0; cy < resolution; cy++)
	for (uint32_t cx = 0; cx < resolution; cx++) {
		int x0, x1, y0, y1, z0, z1;
		voxelRange(cx, resolution, grid.nx, x0, x1);
		voxelRange(cy, resolution, grid.ny, y0, y1);
		voxelRange(cz, resolution, grid.nz, z0, z1);

		float maxDensity = 0.0f;
		for (int z = z0; z <= z1; z++)
		for (int y = y0; y <= y1; y++)
		for (int x = x0; x <= x1; x++)
			maxDensity = (std::max)(maxDensity, grid.density[(size_t(z) * grid.ny + y) * grid.nx + x]);

		grid.majorants[(size_t(cz) * resolution + cy) * resolution + cx] = maxDensity;
	}
}

}
