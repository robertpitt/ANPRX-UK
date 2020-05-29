#!/bin/bash
version=4.2.0
base_url=https://github.com/opencv/opencv/releases/download/${version}/

# Remove old SDK
rm -rf ./vendor/OpenCV-android-sdk/

# Download SDK (dont redownload unless required.)
if ! [ -f opencv-${version}-android-sdk.zip ]; then
  curl -L ${base_url}/opencv-${version}-android-sdk.zip -o opencv-${version}-android-sdk.zip
fi

# Extract
mkdir vendor
unzip opencv-${version}-android-sdk.zip -d vendor
echo "1) Open setting.gradle file and append these two lines."
echo "  include ':opencv'"
echo "  project(':opencv').projectDir = new File(opencvsdk + '/sdk')"

echo "2) Open build.gradle file and add implementation project(path: ':opencv') to dependencies section :"
echo "3) Click on File -> Sync Project with Gradle Files."

# unzip opencv-${version}-android-sdk.zip
# mkdir -p ./src/main/jniLibs
# cp -r ./OpenCV-android-sdk/sdk/native/libs/ ./app/src/main/jniLibs

# mkdir -p ./libraries/opencv
# cp -r ./OpenCV-android-sdk/sdk/java/** ./libraries/opencv
# rm -rf opencv-${version}-android-sdk.zip
# rm -rf ./OpenCV-android-sdk/

# echo '1) Sync the gradle project.\n2) Import the libraries/opencv via File > New > Import Module'