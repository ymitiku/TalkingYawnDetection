#!/bin/bash 
dataset_path= "~/datasets/Yaw/YawDD\ dataset/Mirror/Male_mirror/";
output_path="/dataset/yawn/images3/"

for file in $dataset_path*.avi; 
do 
filename="$(basename "${file}" .avi)"
mkdir "$output_path/${filename}"
ffmpeg -i "$file" "$output_path/${filename}/${output-%05d}".jpg;
done;

for file in ~/datasets/Yaw/YawDD\ dataset/Mirror/Female_mirror/*.avi; 
do 
filename="$(basename "${file}" .avi)"
mkdir "$output_path/${filename}"
ffmpeg -i "$file" "$output_path/${filename}/${output-%05d}".jpg;
done;