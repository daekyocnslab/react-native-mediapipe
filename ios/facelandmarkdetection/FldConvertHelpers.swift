import Foundation
import MediaPipeTasksVision
import React

func convertFldResultBundleToDictionary(_ resultBundle: FaceLandmarkDetectionResultBundle) -> [String: Any] {
    var map = [String: Any]()
    
    // Basic fields
    map["inferenceTime"] = resultBundle.inferenceTime
    map["inputImageHeight"] = Int(resultBundle.size.height)
    map["inputImageWidth"] = Int(resultBundle.size.width)
    map["inputImageRotation"] = 0 // iOS doesn't track rotation the same way
    
    // Cropped frame (Base64 image)
    if let croppedFrame = resultBundle.croppedFrame {
        let base64String = croppedFrame.base64EncodedString()
        map["croppedFrame"] = base64String
        print("✅ Added croppedFrame: \(croppedFrame.count) bytes -> \(base64String.count) base64 chars")
    } else {
        print("❌ croppedFrame is NULL")
    }
    
    // ONNX input data
    if let onnxInputData = resultBundle.onnxInputData {
        let base64String = onnxInputData.base64EncodedString()
        map["onnxInputData"] = base64String
        print("✅ Added onnxInputData: \(onnxInputData.count) bytes -> \(base64String.count) base64 chars")
    } else {
        print("❌ onnxInputData is NULL")
    }
    
    // Face features data
    if let faceFeatures = resultBundle.faceFeatures {
        var featuresMap = [String: Any]()
        featuresMap["earLeft"] = Double(faceFeatures.earLeft)
        featuresMap["earRight"] = Double(faceFeatures.earRight)
        featuresMap["earAvg"] = Double(faceFeatures.earAvg)
        featuresMap["gazeLeft"] = Double(faceFeatures.gazeLeft)
        featuresMap["gazeRight"] = Double(faceFeatures.gazeRight)
        featuresMap["mar"] = Double(faceFeatures.mar)
        featuresMap["leftEyeX"] = Double(faceFeatures.leftEyeX)
        featuresMap["leftEyeY"] = Double(faceFeatures.leftEyeY)
        featuresMap["rightEyeX"] = Double(faceFeatures.rightEyeX)
        featuresMap["rightEyeY"] = Double(faceFeatures.rightEyeY)
        featuresMap["leftIrisX"] = Double(faceFeatures.leftIrisX)
        featuresMap["leftIrisY"] = Double(faceFeatures.leftIrisY)
        featuresMap["rightIrisX"] = Double(faceFeatures.rightIrisX)
        featuresMap["rightIrisY"] = Double(faceFeatures.rightIrisY)
        featuresMap["faceCenterX"] = Double(faceFeatures.faceCenterX)
        featuresMap["faceCenterY"] = Double(faceFeatures.faceCenterY)
        featuresMap["leftEyeWidth"] = Double(faceFeatures.leftEyeWidth)
        featuresMap["rightEyeWidth"] = Double(faceFeatures.rightEyeWidth)
        
        map["faceFeatures"] = featuresMap
        print("✅ Added faceFeatures")
    } else {
        print("❌ faceFeatures is NULL")
    }
    
    // Results array processing
    let resultsArray = resultBundle.faceLandmarkDetectorResults.map { result -> [String: Any] in
        var resultMap = [String: Any]()
        
        if let result = result {
            // Landmarks processing
            let landmarksArray = result.faceLandmarks.map { facelandmarks -> [[String: Any]] in
                return facelandmarks.map { landmark -> [String: Any] in
                    return [
                        "x": Double(landmark.x),
                        "y": Double(landmark.y),
                        "z": Double(landmark.z)
                    ]
                }
            }
            resultMap["landmarks"] = landmarksArray
            
            // Blendshapes processing (if available)
            var blendshapesArray = [[[String: Any]]]()
            if !result.faceBlendshapes.isEmpty {
                blendshapesArray = result.faceBlendshapes.map { faceBlendshapes -> [[String: Any]] in
                    return faceBlendshapes.categories.map { blendshape -> [String: Any] in
                        return [
                            "categoryName": blendshape.categoryName ?? "",
                            "score": Double(blendshape.score)
                        ]
                    }
                }
            }
            resultMap["faceBlendshapes"] = blendshapesArray
            
            // Other fields
            resultMap["worldLandmarks"] = []
            resultMap["segmentationMask"] = []
            resultMap["facialTransformationMatrixes"] = []
            
        } else {
            // Handle nil result
            resultMap["landmarks"] = []
            resultMap["faceBlendshapes"] = []
            resultMap["worldLandmarks"] = []
            resultMap["segmentationMask"] = []
            resultMap["facialTransformationMatrixes"] = []
        }
        
        return resultMap
    }
    
    map["results"] = resultsArray
    
    print("=== convertFldResultBundleToDictionary 완료 ===")
    return map
}
