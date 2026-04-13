#pragma once
// Hilbert curve tile schedule for persistent kernel SM assignment.
// Pure host code — no CUDA dependencies.

#include <algorithm>
#include <cassert>
#include <cstring>
#include <string>
#include <vector>

inline void hilbert_rot(int n, int &x, int &y, int rx, int ry) {
  if (ry == 0) {
    if (rx == 1) {
      x = n - 1 - x;
      y = n - 1 - y;
    }
    int t = x;
    x = y;
    y = t;
  }
}

inline void hilbert_d2xy(int n, int d, int &x, int &y) {
  int rx, ry, s, t = d;
  x = y = 0;
  for (s = 1; s < n; s *= 2) {
    rx = 1 & (t / 2);
    ry = 1 & (t ^ rx);
    hilbert_rot(s, x, y, rx, ry);
    x += s * rx;
    y += s * ry;
    t /= 4;
  }
}

// Build a Hilbert-curve-ordered tile schedule.
// M, N:       number of tiles in each dimension
// CORES:      number of clusters (each gets a slot list)
// SPACE_LEN:  max tiles per cluster
// space:      output array of size CORES * SPACE_LEN, filled with
//             (tile_m << 16 | tile_n) entries, -1 terminated.
inline void createHilbert(int M, int N, int CORES, int SPACE_LEN, int *space) {
  int dim = (1 << (32 - __builtin_clz(std::max(M, N) - 1)));
  int core = 0;
  std::vector<std::string> v(dim, std::string(dim, '.'));
  memset(space, -1, sizeof(int) * CORES * SPACE_LEN);
  int FCORES = 64;
  if (FCORES > CORES)
    FCORES = CORES;
  int total = 0;
  std::vector<std::vector<int>> pos(CORES, std::vector<int>());
  for (int i = 0; i < dim * dim; ++i) {
    int x, y;
    hilbert_d2xy(dim, i, x, y);
    if (x < M && y < N) {
      assert((int)pos[core].size() < SPACE_LEN);
      assert(v[x][y] == '.');
      v[x][y] = '*';
      ++total;
      pos[core].push_back((x << 16) | y);
      ++core;
      if (core == FCORES) {
        core = 0;
      }
    }
  }
  core = FCORES;
  for (int i = 0; i < FCORES; ++i) {
    if (pos.back().size() >= pos[0].size() - 1)
      break;
    pos[core].push_back(pos[i].back());
    pos[i].pop_back();
    ++core;
    if (core == CORES) {
      core = FCORES;
    }
  }
  for (int i = 0; i < CORES; ++i) {
    for (int j = 0; j < (int)pos[i].size(); ++j) {
      space[i * SPACE_LEN + j] = pos[i][j];
    }
  }
  assert(total == M * N);
}
