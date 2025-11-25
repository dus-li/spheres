<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- SPDX-FileCopyrightText: Dus'li -->

# Sphere Casting

## Context

This is a university project for a subject on CUDA programming. Before reading
it (I'm really sorry if you have to) you need to know, that my limited
experience with C++ is only eclipsed by my even poorer knowledge of GPU
programming.

But that's kind of the point of studying? You know, to learn, even if the
intermediate state is not exceptionally pretty. So yeah, here goes nothing.

## What this does

It randomizes a collection of spheres of different sizes and materials in a
space and places point light sources of various randomized parameters
surrounding the blob of spheres. Then it casts rays and renders the scene using
Phong illumination model with few additional sprinkles here and there to make
the result a wee bit easier on the eye.

## Navigating the project

``` text
├ 📁 LICENSES    Licenses, both for my stuff and others' stuff that's here
├ 📁 include
│ ├ 📁 device    CUDA headers
│ └ 📁 host      Pure C++ headers
├ 📁 resources   Font for printing frames per second
├ 📁 scripts
│ └ 📁 hooks     Prevents me from pushing something stupid to remote
├ 📁 source
│ ├ 📁 device    CUDA source files
│ └ 📁 host      Pure C++ source files
├ .clang-format  Aids in cleaning up the formatting
└ Makefile       Helps with building the thing
```

Highlights of most interesting places:

- `sources/main.cu` - Root of all evil resides here.
- `sources/device/scene.cu` - At the bottom of the file rendering happens.
- `include/device/vecops.cuh` - This is where I reinvent the wheel.

## What is needed to build this

Apart from the CUDA toolchain, just SDL2 library to handle managing a window.

I have only tested this on a GNU/Linux machine. Thanks to AI being a thing now,
it turns out that NVIDIA compatibility got way better than I remembered it from
last time. I don't recall doing anything Linux-specific in code, but the
Makefile will not work for Windows, unless its ran from like a WSL or Cygwin or
whatnot.
