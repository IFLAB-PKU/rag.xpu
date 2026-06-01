#!/bin/bash

function clean() {
    set +e
    rm -rf /qnn
}

set -e
trap clean EXIT

if [ -f "${HOME}/.bashrc" ]; then
    source "${HOME}/.bashrc"
fi

if [ -f /lib/qnn/unzip_qnn.sh ]; then
    bash /lib/qnn/unzip_qnn.sh
    source /qnn/bin/envsetup.sh
elif [ -n "${QNN_SDK:-}" ] && [ -f "${QNN_SDK}/bin/envsetup.sh" ]; then
    source "${QNN_SDK}/bin/envsetup.sh"
elif [ -n "${QNN_SDK_ROOT:-}" ] && [ -f "${QNN_SDK_ROOT}/bin/envsetup.sh" ]; then
    source "${QNN_SDK_ROOT}/bin/envsetup.sh"
else
    echo -e "\033[31mQNN envsetup.sh not found\033[0m"
    exit 1
fi

if [ -n "${PYTHON_VENV_PATH:-}" ] && [ -f "${PYTHON_VENV_PATH}/bin/activate" ]; then
    source "${PYTHON_VENV_PATH}/bin/activate"
fi

ANDROID_NDK=${ANDROID_NDK:-/ndk}
echo -e "\033[32mSetting up NDK environment variable\033[0m"
if [ -z "$ANDROID_NDK" ] || [ ! -d "$ANDROID_NDK" ]; then
    echo -e "\033[31mNDK not found\033[0m"
    exit 1
else
    echo -e "\033[32mNDK found at $ANDROID_NDK\033[0m"
fi

if [ -z "$QNN_SDK_ROOT" ]; then
    echo -e "\033[31mQNN_SDK_ROOT not found\033[0m"
    exit 1
else
    echo -e "\033[32mQNN_SDK_ROOT found at $QNN_SDK_ROOT\033[0m"
fi

WORKSPACE=${WORKSPACE:-/workspace}
if [ ! -d "$WORKSPACE" ]; then
    WORKSPACE=/code
fi
cd "$WORKSPACE"

echo -e "\033[32mCreating build directory for Android\033[0m"
cmake -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK/build/cmake/android.toolchain.cmake -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=android-34 -DCMAKE_BUILD_TYPE=RelWithDebInfo -DBUILD_SHARED_LIBS=OFF -DGGML_OPENMP=OFF -DPOWERSERVE_WITH_QNN=OFF -DPOWERSERVE_ENABLE_HTPRPCPOLL=OFF -DPOWERSERVE_ENABLE_HMXPWRCFG=OFF -DPOWERSERVE_USE_DUMMY=ON -DPOWERSERVE_WITH_OPENCL=OFF -DPOWERSERVE_OPENCL_EMBED_KERNELS=OFF -DPOWERSERVE_ANDROID_OPENCL_LIB="${WORKSPACE}/third_party/android/arm64-v8a/libOpenCL.so" -DOpenCL_LIBRARY="${WORKSPACE}/third_party/android/arm64-v8a/libOpenCL.so" -DOpenCL_INCLUDE_DIR="${ANDROID_NDK}/toolchains/llvm/prebuilt/linux-x86_64/sysroot/usr/include" -S . -B build_android

echo -e "\033[32mBuilding project for Android\033[0m"
cmake --build build_android --config RelWithDebInfo --parallel 12 --target all

tempmodel1="/tempA"
tempmodel2="/tempB"
mkdir -p "${tempmodel1}"
mkdir -p "${tempmodel2}"
touch "${tempmodel1}/model.json"
touch "${tempmodel2}/model.json"

./powerserve create --only-extract-qnn -m "${tempmodel1}" -d "${tempmodel2}" --exe-path "${WORKSPACE}/build_android/out"

rm -rf "${tempmodel1}" || true
rm -rf "${tempmodel2}" || true

chmod -R 777 "${WORKSPACE}/proj"
chmod -R 777 "${WORKSPACE}/build_android"
mkdir -p /models
chmod 777 /models
chmod 777 ./*.log 2>/dev/null || true
