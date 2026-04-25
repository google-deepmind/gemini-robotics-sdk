// This file acts as a single compilation unit for the mcap library.
// This resolves linker errors one-definition-rule violations.

// This define is the key. It tells the mcap headers to include the
// actual function bodies.
#ifndef MCAP_IMPLEMENTATION
#define MCAP_IMPLEMENTATION
#endif

// Include the primary headers for the mcap library.
// Since MCAP_IMPLEMENTATION is defined, the preprocessor will now
// include the implementation code from within these headers or
// any .inl files they include.
#include "mcap/mcap.hpp"
