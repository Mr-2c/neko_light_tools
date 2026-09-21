Real Neko output for the permute_fields.py tests (Neko 0.9-dev built with
gfortran/OpenMPI, json-fortran 8.3): the Taylor-Green vortex user case
(tgv.f90 from Neko's examples) on a 4x4x4 periodic box, polynomial order 3,
five steps of dt = 0.01, double-precision output.

* `box64.nmsh` -- genmeshbox 0..2pi cube, 4x4x4, periodic in x, y, z
* `box64_g211.nmsh` -- the same mesh re-ordered by `prepart.py --grid 2,1,1`
* `A/` -- run on box64.nmsh with 1 rank: `field0.f00000` (t = 0, with
  coordinates), `field0.f00001` (step 5), `fluid00001.chkp` (step 5),
  `field0.nek5000`, and the case file
* `B/` -- the same run on box64_g211.nmsh with 2 ranks

The two runs differ only by the parallel gather-scatter round-off and the
iterative solver tolerances, so `permute_fields.py box64.nmsh
box64_g211.nmsh -o out A/*` must reproduce the files in `B/` element for
element (coordinates, pressure and idx exactly at t = 0).
