# Mouth Open Detector (MOD)

This repository contains the code for the research article **"Mouth Detector with Combination Between LipNet and PyImageSearch"**, published in the World Journal of Advanced Research and Reviews (WJARR), Volume 22, Issue 3, 2024. The study explores the innovative integration of LipNet and PyImageSearch to develop a robust mouth detection system using deep learning and computer vision, aimed at enhancing the accuracy and reliability of automatic lip reading systems.

## Introduction

In this study, we delve into the innovative integration of LipNet and PyImageSearch to develop a robust mouth detection system. Our approach harnesses the strengths of deep learning and computer vision, aiming to enhance the accuracy and reliability of automatic lip reading systems. This advancement holds significant promise for applications in accessibility, security, and multimedia.

## Usage

To run the mouth open detector, use the following commands:

```sh
# Basic usage
python detect_drowsiness.py --shape-predictor shape_predictor_68_face_landmarks.dat

# With alarm
python detect_drowsiness.py --shape-predictor shape_predictor_68_face_landmarks.dat --alarm alarm.wav
