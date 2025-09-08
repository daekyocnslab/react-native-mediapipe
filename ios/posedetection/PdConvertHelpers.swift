import Foundation
import MediaPipeTasksVision
import React

// PoseMap 구조체 추가 (Kotlin의 PoseMap.POSE_LANDMARKS에 해당)
struct PoseMap {
  static let POSE_LANDMARKS: [Int: String] = [
    0: "nose",
    15: "left_wrist",
    16: "right_wrist",
    19: "left_index",
    20: "right_index"
  ]
}

func convertPdResultBundleToDictionary(_ resultBundle: PoseDetectionResultBundle) -> [String: Any] {
  var map = [String: Any]()
  
  // Results
  let resultsArray = resultBundle.poseDetectorResults.map { result -> [String: Any] in
    var resultMap = [String: Any]()
    resultMap["timestampMs"] = result?.timestampInMilliseconds
    
    let landmarks = result?.landmarks.map { $0.map(normalizedLandmarkToDictionary) }
    let worldLandmarks = result?.worldLandmarks.map { $0.map(landmarkToDictionary) }
    
    // 선택된 랜드마크 처리 추가
    let selectedLandmarks = result?.landmarks.map { landmarkGroup -> [String: Any] in
      var selectedLandmarkMap = [String: Any]()
      
      for (index, landmark) in landmarkGroup.enumerated() {
        if let landmarkName = PoseMap.POSE_LANDMARKS[index] {
          var landmarkData = [String: Any]()
          
          // visibility가 0.5 이상일 때만 좌표 저장 (Python 코드와 동일)
            if let visibility = landmark.visibility, visibility.doubleValue > 0.5 {
            // normalized 좌표에 이미지 크기를 곱해서 실제 픽셀 좌표로 변환
            let actualX = landmark.x * Float(resultBundle.size.width)
            let actualY = landmark.y * Float(resultBundle.size.height)
            
            landmarkData["x"] = Double(actualX)
            landmarkData["y"] = Double(actualY)
            landmarkData["visibility"] = visibility.doubleValue
          } else {
            // visibility가 0.5 이하일 때는 null 저장 (Python에서 np.nan과 동일)
            landmarkData["x"] = NSNull()
            landmarkData["y"] = NSNull()
//            landmarkData["visibility"] = Double(landmark.visibility)
            landmarkData["visibility"] = NSNull()
          }
          
          selectedLandmarkMap[landmarkName] = landmarkData
        }
      }
      
      return selectedLandmarkMap
    }
    
    // this is typically a float for every frame. Too much. It can never go over the boundary
//    let segmentationMasks = result?.segmentationMasks.map { maskToDictionary($0) }
    
    return [
      "landmarks": landmarks ?? [],
      "worldLandmarks": worldLandmarks ?? [],
      "segmentationMasks": [],
      "selectedLandmarks": selectedLandmarks ?? [] // 선택된 랜드마크 추가
    ]
  }
  map["results"] = resultsArray
  
  // Image properties
  map["inputImageHeight"] = resultBundle.size.height
  map["inputImageWidth"] = resultBundle.size.width
  //  map["inputImageRotation"] = resultBundle.
  map["inferenceTime"] = resultBundle.inferenceTime
  
  return map
}
