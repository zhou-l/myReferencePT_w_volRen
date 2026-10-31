#pragma once

#include "Structures.h"

// CPU-side creation of the density grid used by the participating medium
namespace Volume
{
	// Loads a Mitsuba 0.6 grid volume (.vol, float32 encoding). Uses the first channel.
	bool LoadVol(const std::string& filename, DensityGrid& grid);

	// Fills the grid with a procedural cloud (noise-perturbed sphere), values in [0, 1]
	void GenerateCloud(DensityGrid& grid, uint32_t resolution);

	// Builds the coarse majorant grid: each cell stores the max density that trilinear
	// interpolation can return inside that cell
	void BuildMajorants(DensityGrid& grid, uint32_t resolution);
}
