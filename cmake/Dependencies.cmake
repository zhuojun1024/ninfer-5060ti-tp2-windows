find_package(CUDAToolkit REQUIRED)
find_package(Threads REQUIRED)

# Media decoding needs libavformat/libavcodec/libavutil/libswscale. The maintainer environment exposes
# them through pkg-config; a native Windows checkout has no pkg-config, so the same four libraries are
# located from an explicit prefix instead of a system-wide search that could pick up an unrelated
# toolchain.
add_library(ninfer::ffmpeg INTERFACE IMPORTED GLOBAL)
if(WIN32)
  set(NINFER_FFMPEG_ROOT "" CACHE PATH
      "Prefix holding the FFmpeg include/ and lib/ trees used by the Windows build")
  if(NOT NINFER_FFMPEG_ROOT)
    message(FATAL_ERROR
      "NINFER_FFMPEG_ROOT must name an FFmpeg prefix (include/ and lib/); for example "
      "-DNINFER_FFMPEG_ROOT=C:/ffmpeg")
  endif()
  find_path(NINFER_FFMPEG_INCLUDE_DIR NAMES libavcodec/avcodec.h
            PATHS "${NINFER_FFMPEG_ROOT}/include" NO_DEFAULT_PATH REQUIRED)
  target_include_directories(ninfer::ffmpeg INTERFACE "${NINFER_FFMPEG_INCLUDE_DIR}")
  foreach(component avformat avcodec avutil swscale)
    find_library(NINFER_FFMPEG_${component}_LIBRARY NAMES ${component} lib${component}
                 PATHS "${NINFER_FFMPEG_ROOT}/lib" NO_DEFAULT_PATH REQUIRED)
    target_link_libraries(ninfer::ffmpeg INTERFACE "${NINFER_FFMPEG_${component}_LIBRARY}")
  endforeach()
else()
  find_package(PkgConfig REQUIRED)
  pkg_check_modules(FFMPEG REQUIRED IMPORTED_TARGET
    libavformat libavcodec libavutil libswscale)
  target_link_libraries(ninfer::ffmpeg INTERFACE PkgConfig::FFMPEG)
endif()

# Repository-pinned header dependencies. No configure-time downloads.
add_library(ninfer::json INTERFACE IMPORTED GLOBAL)
target_include_directories(ninfer::json INTERFACE
  ${PROJECT_SOURCE_DIR}/third_party)

# Source base for the custom-template frontend; consumers will link it explicitly.
add_subdirectory(third_party/llama-jinja EXCLUDE_FROM_ALL)

if(NINFER_BUILD_PRODUCT_SUPPORT)
  # Media acquisition uses CURLOPT_PROTOCOLS_STR and CURLOPT_REDIR_PROTOCOLS_STR,
  # introduced in libcurl 7.85 (not merely the version of the maintainer environment).
  add_library(ninfer::curl INTERFACE IMPORTED GLOBAL)
  if(WIN32)
    set(NINFER_LIBCURL_ROOT "" CACHE PATH
        "Prefix holding the libcurl include/ and lib/ trees used by the Windows build")
    if(NOT NINFER_LIBCURL_ROOT)
      message(FATAL_ERROR
        "NINFER_LIBCURL_ROOT must name a libcurl prefix (include/ and lib/); for example "
        "-DNINFER_LIBCURL_ROOT=C:/curl")
    endif()
    find_path(NINFER_LIBCURL_INCLUDE_DIR NAMES curl/curl.h
              PATHS "${NINFER_LIBCURL_ROOT}/include" NO_DEFAULT_PATH REQUIRED)
    # A Windows prefix can ship both a static archive and an import library. The import library is
    # required here: it keeps libcurl's own dependencies (TLS, compression) inside the DLL, and MSVC
    # cannot link a static mingw archive in the first place.
    find_file(NINFER_LIBCURL_IMPORT_LIBRARY NAMES libcurl.dll.a curl.dll.a
              PATHS "${NINFER_LIBCURL_ROOT}/lib" NO_DEFAULT_PATH)
    if(NINFER_LIBCURL_IMPORT_LIBRARY)
      set(NINFER_LIBCURL_LIBRARY "${NINFER_LIBCURL_IMPORT_LIBRARY}" CACHE FILEPATH
          "libcurl import library used by the Windows build" FORCE)
    else()
      find_library(NINFER_LIBCURL_LIBRARY NAMES curl libcurl
                   PATHS "${NINFER_LIBCURL_ROOT}/lib" NO_DEFAULT_PATH REQUIRED)
    endif()
    target_include_directories(ninfer::curl INTERFACE "${NINFER_LIBCURL_INCLUDE_DIR}")
    target_link_libraries(ninfer::curl INTERFACE "${NINFER_LIBCURL_LIBRARY}")
  else()
    pkg_check_modules(LIBCURL REQUIRED IMPORTED_TARGET libcurl>=7.85)
    target_link_libraries(ninfer::curl INTERFACE PkgConfig::LIBCURL)
  endif()
  add_library(ninfer::httplib INTERFACE IMPORTED GLOBAL)
  target_include_directories(ninfer::httplib INTERFACE
    ${PROJECT_SOURCE_DIR}/third_party/cpp-httplib)
  add_subdirectory(third_party/spdlog)
endif()
