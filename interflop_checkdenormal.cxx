
/*--------------------------------------------------------------------*/
/*--- Verrou: a FPU instrumentation tool.                          ---*/
/*--- Interface for floating-point operations overloading.         ---*/
/*---                                                 vr_fpOps.cxx ---*/
/*--------------------------------------------------------------------*/

/*
   This file is part of Verrou, a FPU instrumentation tool.

   Copyright (C) 2014-2021 EDF
     F. Févotte     <francois.fevotte@edf.fr>
     B. Lathuilière <bruno.lathuiliere@edf.fr>

   This program is free software; you can redistribute it and/or
   modify it under the terms of the GNU Lesser General Public License as
   published by the Free Software Foundation; either version 2.1 of the
   License, or (at your option) any later version.

   This program is distributed in the hope that it will be useful, but
   WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
   General Public License for more details.

   You should have received a copy of the GNU Lesser General Public License
   along with this program; if not, write to the Free Software
   Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA
   02111-1307, USA.

   The GNU Lesser General Public License is contained in the file COPYING.
*/

#include <argp.h>
#include <cmath>
#include <limits>
#include <stddef.h>

#include "interflop/fma/interflop_fma.h"
#include "interflop/interflop.h"
#include "interflop/interflop_stdlib.h"
#include "interflop/iostream/logger.h"
#include "interflop_checkdenormal.h"

// * Global variables & parameters

static const char backend_name[] = "interflop-checkdenormal";
static const char backend_version[] = "1.x-dev";

typedef enum { KEY_FTZ } key_args;

static const char key_ftz_str[] = "flush-to-zero";

static File *stderr_stream;

template <typename REAL>
void ifcd_checkdenorm(const REAL &a, const REAL &b, const REAL &r);

template <class REAL>
void flushToZeroAndCheck(REAL *res, checkdenormal_context_t *ctx) {
  if (std::abs(*res) < std::numeric_limits<REAL>::min() && *res != 0.) {
    if (interflop_denormalHandler != Null) {
      interflop_denormalHandler();
    }
    if (ctx->flushtozero) {
      *res = 0.;
    }
  }
}

#if defined(__cplusplus)
extern "C" {
#endif

// * C interface

static void _set_checkdenormal_ftz(bool ftz, checkdenormal_context_t *ctx) {
  ctx->flushtozero = ftz;
}

#ifdef IFCD_DOOP
#define APPLYOP(a, b, res, op)                                                 \
  *res = a op b;                                                               \
  flushToZeroAndCheck(res, ctx);
#else
#define APPLYOP(a, b, res, op) flushToZeroAndCheck(res, ctx);
#endif

void INTERFLOP_CHECKDENORMAL_API(add_double)(double a, double b, double *res,
                                             void *context) {
  checkdenormal_context_t *ctx = (checkdenormal_context_t *)context;
  APPLYOP(a, b, res, +);
}

void INTERFLOP_CHECKDENORMAL_API(add_float)(float a, float b, float *res,
                                            void *context) {
  checkdenormal_context_t *ctx = (checkdenormal_context_t *)context;
  APPLYOP(a, b, res, +);
}

void INTERFLOP_CHECKDENORMAL_API(sub_double)(double a, double b, double *res,
                                             void *context) {
  checkdenormal_context_t *ctx = (checkdenormal_context_t *)context;
  APPLYOP(a, b, res, -);
}

void INTERFLOP_CHECKDENORMAL_API(sub_float)(float a, float b, float *res,
                                            void *context) {
  checkdenormal_context_t *ctx = (checkdenormal_context_t *)context;
  APPLYOP(a, b, res, -);
}

void INTERFLOP_CHECKDENORMAL_API(mul_double)(double a, double b, double *res,
                                             void *context) {
  checkdenormal_context_t *ctx = (checkdenormal_context_t *)context;
  APPLYOP(a, b, res, *);
}

void INTERFLOP_CHECKDENORMAL_API(mul_float)(float a, float b, float *res,
                                            void *context) {
  checkdenormal_context_t *ctx = (checkdenormal_context_t *)context;
  APPLYOP(a, b, res, *);
}

void INTERFLOP_CHECKDENORMAL_API(div_double)(double a, double b, double *res,
                                             void *context) {
  checkdenormal_context_t *ctx = (checkdenormal_context_t *)context;
  APPLYOP(a, b, res, /);
}

void INTERFLOP_CHECKDENORMAL_API(div_float)(float a, float b, float *res,
                                            void *context) {
  checkdenormal_context_t *ctx = (checkdenormal_context_t *)context;
  APPLYOP(a, b, res, /);
}

void INTERFLOP_CHECKDENORMAL_API(fma_float)(float a, float b, float c,
                                            float *res, void *context) {
#ifdef IFCD_DOOP
  *res = interflop_fma_binary32(a, b, c);
  checkdenormal_context_t *ctx = (checkdenormal_context_t *)context;
  flushToZeroAndCheck(res, ctx);
#endif
}

void INTERFLOP_CHECKDENORMAL_API(fma_double)(double a, double b, double c,
                                             double *res, void *context) {
#ifdef IFCD_DOOP
  *res = interflop_fma_binary64(a, b, c);
  checkdenormal_context_t *ctx = (checkdenormal_context_t *)context;
  flushToZeroAndCheck(res, ctx);
#endif
}

void INTERFLOP_CHECKDENORMAL_API(cast_double_to_float)(double a, float *res,
                                                       void *context) {
#ifdef IFCD_DOOP
  *res = (float)a;
#endif
  checkdenormal_context_t *ctx = (checkdenormal_context_t *)context;
  flushToZeroAndCheck(res, ctx);
}

void INTERFLOP_CHECKDENORMAL_API(finalize)([[maybe_unused]] void *context) {}

const char *INTERFLOP_CHECKDENORMAL_API(get_backend_name)() {
  return backend_name;
}

const char *INTERFLOP_CHECKDENORMAL_API(get_backend_version)() {
  return backend_version;
}

void _checkdenormal_check_stdlib(void) {
  INTERFLOP_CHECK_IMPL(denormalHandler);
  INTERFLOP_CHECK_IMPL(malloc);
  INTERFLOP_CHECK_IMPL(strcasecmp);
  INTERFLOP_CHECK_IMPL(strtol);
}

void _checkdenormal_alloc_context(void **context) {
  *context = (checkdenormal_context_t *)interflop_malloc(
      sizeof(checkdenormal_context_t));
}

void _checkdenormal_init_context(checkdenormal_context_t *ctx) {
  ctx->flushtozero = IFalse;
}

void INTERFLOP_CHECKDENORMAL_API(pre_init)(interflop_panic_t panic,
                                           File *stream, void **context) {
  stderr_stream = stream;
  interflop_set_handler("panic", (void *)panic);
  _checkdenormal_check_stdlib();

  /* Initialize the logger */
  logger_init(panic, stream, backend_name);

  /* Allocate context */
  _checkdenormal_alloc_context(context);
  _checkdenormal_init_context((checkdenormal_context_t *)*context);
}

static struct argp_option end_option = {0, 0, 0, 0, 0, 0};

static struct argp_option options[] = {
    {key_ftz_str, KEY_FTZ, "FTZ", 0, "enable flush-to-zero", 0}, end_option};

static error_t parse_opt(int key, [[maybe_unused]] char *arg,
                         struct argp_state *state) {
  checkdenormal_context_t *ctx = (checkdenormal_context_t *)state->input;
  switch (key) {
  case KEY_FTZ:
    /* flust-to-zero */
    _set_checkdenormal_ftz(ITrue, ctx);
    break;
  default:
    return ARGP_ERR_UNKNOWN;
  }
  return 0;
}

static struct argp argp = {options, parse_opt, "", "", NULL, NULL, NULL};

void INTERFLOP_CHECKDENORMAL_API(cli)(int argc, char **argv, void *context) {
  checkdenormal_context_t *ctx = (checkdenormal_context_t *)context;
  if (interflop_argp_parse != NULL) {
    interflop_argp_parse(&argp, argc, argv, 0, 0, ctx);
  } else {
    interflop_panic("Interflop backend error: argp_parse not implemented\n"
                    "Provide implementation or use interflop_configure to "
                    "configure the backend\n");
  }
}

void INTERFLOP_CHECKDENORMAL_API(configure)(void *configure, void *context) {
  checkdenormal_context_t *ctx = (checkdenormal_context_t *)context;
  checkdenormal_conf_t *conf = (checkdenormal_conf_t *)configure;
  ctx->flushtozero = conf->flushtozero;
}

static void print_information_header(void *context) {
  /* Environnement variable to disable loading message */
  char *silent_load_env = interflop_getenv("VFC_BACKENDS_SILENT_LOAD");
  bool silent_load = ((silent_load_env == NULL) ||
                      (interflop_strcasecmp(silent_load_env, "True") != 0))
                         ? false
                         : true;

  if (silent_load)
    return;

  checkdenormal_context_t *ctx = (checkdenormal_context_t *)context;
  logger_info("load backend with:\n");
  logger_info("%s = %s\n", key_ftz_str, ctx->flushtozero ? "true" : "false");
}

struct interflop_backend_interface_t
INTERFLOP_CHECKDENORMAL_API(init)(void *context) {

  checkdenormal_context_t *ctx = (checkdenormal_context_t *)context;
  print_information_header(ctx);

  struct interflop_backend_interface_t interflop_backend_checkdenormal = {
    interflop_add_float : INTERFLOP_CHECKDENORMAL_API(add_float),
    interflop_sub_float : INTERFLOP_CHECKDENORMAL_API(sub_float),
    interflop_mul_float : INTERFLOP_CHECKDENORMAL_API(mul_float),
    interflop_div_float : INTERFLOP_CHECKDENORMAL_API(div_float),
    interflop_cmp_float : Null,
    interflop_add_double : INTERFLOP_CHECKDENORMAL_API(add_double),
    interflop_sub_double : INTERFLOP_CHECKDENORMAL_API(sub_double),
    interflop_mul_double : INTERFLOP_CHECKDENORMAL_API(mul_double),
    interflop_div_double : INTERFLOP_CHECKDENORMAL_API(div_double),
    interflop_cmp_double : Null,
    interflop_cast_double_to_float :
        INTERFLOP_CHECKDENORMAL_API(cast_double_to_float),
    interflop_fma_float : INTERFLOP_CHECKDENORMAL_API(fma_float),
    interflop_fma_double : INTERFLOP_CHECKDENORMAL_API(fma_double),
    interflop_enter_function : Null,
    interflop_exit_function : Null,
    interflop_user_call : Null,
    interflop_finalize : INTERFLOP_CHECKDENORMAL_API(finalize),
  };
  return interflop_backend_checkdenormal;
}

struct interflop_backend_interface_t interflop_init(void *context)
    __attribute__((weak, alias("interflop_checkdenormal_init")));

void interflop_pre_init(interflop_panic_t panic, File *stream, void **context)
    __attribute__((weak, alias("interflop_checkdenormal_pre_init")));

void interflop_cli(int argc, char **argv, void *context)
    __attribute__((weak, alias("interflop_checkdenormal_cli")));

#if defined(__cplusplus)
}
#endif