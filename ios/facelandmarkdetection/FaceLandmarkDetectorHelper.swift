import AVFoundation
import MediaPipeTasksVision
import UIKit

/// This protocol must be adopted by any class that wants to get the detection results of the object detector in live stream mode.
protocol FaceLandmarkDetectorHelperDelegate: AnyObject {
  func faceLandmarkDetectorHelper(
    _ faceLandmarkDetectorHelper: FaceLandmarkDetectorHelper,
    onResults result: FaceLandmarkDetectionResultBundle?,
    error: Error?)
}

// Face features data structure to match Android
struct FaceFeatures {
  let earLeft: Float
  let earRight: Float
  let earAvg: Float
  let gazeLeft: Float
  let gazeRight: Float
  let mar: Float
  let leftEyeX: Float
  let leftEyeY: Float
  let rightEyeX: Float
  let rightEyeY: Float
  let leftIrisX: Float
  let leftIrisY: Float
  let rightIrisX: Float
  let rightIrisY: Float
  let faceCenterX: Float
  let faceCenterY: Float
  let leftEyeWidth: Float
  let rightEyeWidth: Float
}

// Initializes and calls the MediaPipe APIs for detection.
class FaceLandmarkDetectorHelper: NSObject {

  weak var delegate: FaceLandmarkDetectorHelperDelegate?

  var faceLandmarker: FaceLandmarker?
  private(set) var runningMode = RunningMode.image
  private var numFaces: Int
  private var minFaceDetectionConfidence: Float
  private var minFacePresenceConfidence: Float
  private var minTrackingConfidence: Float
  private var optionsDelegate: Delegate
  var modelPath: String
  
  let handle: Int

  private var livestreamImageSize: CGSize = CGSize(width: 0, height: 0)
  private var currentUIImage: UIImage? // Store current frame for cropping

  // MediaPipe landmark indices (matching Android implementation)
  private let leftEyeIndices = [33, 160, 158, 133, 153, 144]
  private let rightEyeIndices = [362, 385, 387, 263, 373, 380]
  private let leftIrisIndices = [468, 469, 470, 471]
  private let rightIrisIndices = [473, 474, 475, 476]
  
  private let mouthIndices = [
    13, 14, 15, 16, 17, 18,
    61, 84, 17, 314, 405, 320, 307, 375, 308, 324, 318,
    78, 95, 88, 178, 87, 14, 317, 402, 318, 324
  ]

  // MARK: - Custom Initializer
  init(
    handle: Int,
    numFaces: Int,
    minFaceDetectionConfidence: Float,
    minFacePresenceConfidence: Float,
    minTrackingConfidence: Float,
    modelName: String,
    delegate: Int,
    runningMode: RunningMode
  ) throws {
    let fileURL = URL(fileURLWithPath: modelName)

    let basename = fileURL.deletingPathExtension().lastPathComponent
    let fileExtension = fileURL.pathExtension
    guard let modelPath = Bundle.main.path(forResource: basename, ofType: fileExtension) else {
      throw NSError(
        domain: "MODEL_NOT_FOUND", code: 0, userInfo: ["message": "Model \(modelName) not found"])
    }
      
      // =================================================================
      // [디버깅 로그 추가] JavaScript에서 전달된 값들을 여기서 확인합니다.
      // =================================================================
      print("🚀 [FaceLandmarkDetectorHelper] Initializing with values:")
      print("   - minFaceDetectionConfidence: \(minFaceDetectionConfidence)")
      print("   - minFacePresenceConfidence: \(minFacePresenceConfidence)")
      print("   - minTrackingConfidence: \(minTrackingConfidence)")
      // =================================================================
      
    self.handle = handle
    self.modelPath = modelPath
    self.optionsDelegate = convertIntToDelegate(delegate)
    self.numFaces = numFaces
    self.minFaceDetectionConfidence = minFaceDetectionConfidence
    self.minFacePresenceConfidence = minFacePresenceConfidence
    self.minTrackingConfidence = minTrackingConfidence
    self.runningMode = runningMode
    super.init()

    createFaceLandmarkDetector()
  }

  private func createFaceLandmarkDetector() {
    let faceLandmarkerOptions = FaceLandmarkerOptions()
    faceLandmarkerOptions.runningMode = self.runningMode
    faceLandmarkerOptions.numFaces = self.numFaces
    faceLandmarkerOptions.minFaceDetectionConfidence = self.minFaceDetectionConfidence
    faceLandmarkerOptions.minFacePresenceConfidence = self.minFacePresenceConfidence
    faceLandmarkerOptions.minTrackingConfidence = self.minTrackingConfidence
    faceLandmarkerOptions.baseOptions.modelAssetPath = self.modelPath
    faceLandmarkerOptions.baseOptions.delegate = self.optionsDelegate
    faceLandmarkerOptions.outputFaceBlendshapes = true
    if runningMode == .liveStream {
      faceLandmarkerOptions.faceLandmarkerLiveStreamDelegate = self
    }
    do {
      faceLandmarker = try FaceLandmarker(options: faceLandmarkerOptions)
    } catch {
      print(error)
    }
  }

  // MARK: - Face Features Calculation
  private func calculateEAR(eyeLandmarks: [NormalizedLandmark]) -> Float {
    guard eyeLandmarks.count == 6 else { return 0.0 }
    
    let verticalDist1 = sqrt(
      pow(eyeLandmarks[1].x - eyeLandmarks[5].x, 2) +
      pow(eyeLandmarks[1].y - eyeLandmarks[5].y, 2)
    )
    let verticalDist2 = sqrt(
      pow(eyeLandmarks[2].x - eyeLandmarks[4].x, 2) +
      pow(eyeLandmarks[2].y - eyeLandmarks[4].y, 2)
    )
    
    let horizontalDist = sqrt(
      pow(eyeLandmarks[0].x - eyeLandmarks[3].x, 2) +
      pow(eyeLandmarks[0].y - eyeLandmarks[3].y, 2)
    )
    
    guard horizontalDist > 0 else { return 0.0 }
    return (verticalDist1 + verticalDist2) / (2.0 * horizontalDist)
  }
  
  private func calculateMAR(landmarks: [NormalizedLandmark]) -> Float {
    let verticalDistances = [
      sqrt(pow(landmarks[13].x - landmarks[14].x, 2) + pow(landmarks[13].y - landmarks[14].y, 2)),
      sqrt(pow(landmarks[78].x - landmarks[95].x, 2) + pow(landmarks[78].y - landmarks[95].y, 2)),
      sqrt(pow(landmarks[308].x - landmarks[324].x, 2) + pow(landmarks[308].y - landmarks[324].y, 2))
    ]
    
    let horizontalDist = sqrt(
      pow(landmarks[61].x - landmarks[291].x, 2) +
      pow(landmarks[61].y - landmarks[291].y, 2)
    )
    
    let avgVerticalDist = verticalDistances.reduce(0, +) / Float(verticalDistances.count)
    guard horizontalDist > 0 else { return 0.0 }
    return avgVerticalDist / horizontalDist
  }
  
  private func estimateGaze(irisCenter: (Float, Float), eyeLeft: NormalizedLandmark, eyeRight: NormalizedLandmark) -> Float {
    let eyeCenterX = (eyeLeft.x + eyeRight.x) / 2.0
    let gazeOffset = irisCenter.0 - eyeCenterX
    let eyeWidth = abs(eyeRight.x - eyeLeft.x)
    guard eyeWidth > 0 else { return 0.0 }
    return gazeOffset / eyeWidth
  }
  
  private func calculateCenter(landmarks: [NormalizedLandmark]) -> (Float, Float) {
    guard !landmarks.isEmpty else { return (0.0, 0.0) }
    let centerX = landmarks.map { $0.x }.reduce(0, +) / Float(landmarks.count)
    let centerY = landmarks.map { $0.y }.reduce(0, +) / Float(landmarks.count)
    return (centerX, centerY)
  }
  
  private func calculateDistance(p1: (Float, Float), p2: (Float, Float)) -> Float {
    return sqrt(pow(p1.0 - p2.0, 2) + pow(p1.1 - p2.1, 2))
  }
  
  private func calculateFaceFeatures(landmarks: [NormalizedLandmark], imageWidth: Int, imageHeight: Int) -> FaceFeatures {
    let requiredLandmarkCount = 478
    guard landmarks.count >= requiredLandmarkCount else {
        print("⚠️ Not enough landmarks to calculate features. Got \(landmarks.count), need \(requiredLandmarkCount). Returning zeroed features.")
        return FaceFeatures(earLeft: 0, earRight: 0, earAvg: 0, gazeLeft: 0, gazeRight: 0, mar: 0, leftEyeX: 0, leftEyeY: 0, rightEyeX: 0, rightEyeY: 0, leftIrisX: 0, leftIrisY: 0, rightIrisX: 0, rightIrisY: 0, faceCenterX: 0, faceCenterY: 0, leftEyeWidth: 0, rightEyeWidth: 0)
    }

    let leftEyeLandmarks = leftEyeIndices.map { landmarks[$0] }
    let rightEyeLandmarks = rightEyeIndices.map { landmarks[$0] }
    let leftIrisLandmarks = leftIrisIndices.map { landmarks[$0] }
    let rightIrisLandmarks = rightIrisIndices.map { landmarks[$0] }
    
    let leftEAR = calculateEAR(eyeLandmarks: leftEyeLandmarks)
    let rightEAR = calculateEAR(eyeLandmarks: rightEyeLandmarks)
    let avgEAR = (leftEAR + rightEAR) / 2.0
    
    let mar = calculateMAR(landmarks: landmarks)
    
    let leftEyeCenter = calculateCenter(landmarks: leftEyeLandmarks)
    let rightEyeCenter = calculateCenter(landmarks: rightEyeLandmarks)
    let leftIrisCenter = calculateCenter(landmarks: leftIrisLandmarks)
    let rightIrisCenter = calculateCenter(landmarks: rightIrisLandmarks)
    
    let faceCenter = calculateCenter(landmarks: landmarks)
    
    let leftGaze = estimateGaze(irisCenter: leftIrisCenter, eyeLeft: landmarks[33], eyeRight: landmarks[133])
    let rightGaze = estimateGaze(irisCenter: rightIrisCenter, eyeLeft: landmarks[362], eyeRight: landmarks[263])
    
    let leftEyePixelLeft = (landmarks[33].x * Float(imageWidth), landmarks[33].y * Float(imageHeight))
    let leftEyePixelRight = (landmarks[133].x * Float(imageWidth), landmarks[133].y * Float(imageHeight))
    let rightEyePixelLeft = (landmarks[362].x * Float(imageWidth), landmarks[362].y * Float(imageHeight))
    let rightEyePixelRight = (landmarks[263].x * Float(imageWidth), landmarks[263].y * Float(imageHeight))
    
    let leftEyeWidth = calculateDistance(p1: leftEyePixelLeft, p2: leftEyePixelRight)
    let rightEyeWidth = calculateDistance(p1: rightEyePixelLeft, p2: rightEyePixelRight)
    
    return FaceFeatures(
      earLeft: leftEAR,
      earRight: rightEAR,
      earAvg: avgEAR,
      gazeLeft: leftGaze,
      gazeRight: rightGaze,
      mar: mar,
      leftEyeX: leftEyeCenter.0 * Float(imageWidth),
      leftEyeY: leftEyeCenter.1 * Float(imageHeight),
      rightEyeX: rightEyeCenter.0 * Float(imageWidth),
      rightEyeY: rightEyeCenter.1 * Float(imageHeight),
      leftIrisX: leftIrisCenter.0 * Float(imageWidth),
      leftIrisY: leftIrisCenter.1 * Float(imageHeight),
      rightIrisX: rightIrisCenter.0 * Float(imageWidth),
      rightIrisY: rightIrisCenter.1 * Float(imageHeight),
      faceCenterX: faceCenter.0 * Float(imageWidth),
      faceCenterY: faceCenter.1 * Float(imageHeight),
      leftEyeWidth: leftEyeWidth,
      rightEyeWidth: rightEyeWidth
    )
  }
    
    private func resizeImage(image: UIImage, targetSize: CGSize) -> UIImage? {
        print("🔄 Resizing image from \(image.size) to \(targetSize)")
        print("🔍 Original image scale: \(image.scale)")
        
        // scale을 1.0으로 명시적으로 설정
        let format = UIGraphicsImageRendererFormat()
        format.scale = 1.0  // 중요: scale을 1.0으로 고정
        format.opaque = false
        
        let renderer = UIGraphicsImageRenderer(size: targetSize, format: format)
        
        let resizedImage = renderer.image { context in
            image.draw(in: CGRect(origin: .zero, size: targetSize))
        }
        
        print("✅ Resized image size: \(resizedImage.size), scale: \(resizedImage.scale)")
        
        // CGImage 크기도 확인
        if let cgImage = resizedImage.cgImage {
            print("🔍 CGImage dimensions: \(cgImage.width)x\(cgImage.height)")
        }
        
        return resizedImage
    }

  // MARK: - Image Processing
  
    private func convertImageToOnnxInput(image: UIImage) -> Data? {
        print("🔍 ONNX Input: Starting conversion for image size: \(image.size), scale: \(image.scale)")
        
        guard let resizedImage = resizeImage(image: image, targetSize: CGSize(width: 64, height: 64)) else {
            print("❌ ONNX Conversion failed: Could not resize to 64x64")
            return nil
        }
        
        print("✅ ONNX Input: Successfully resized to: \(resizedImage.size), scale: \(resizedImage.scale)")
        
        guard let cgImage = resizedImage.cgImage else {
            print("❌ ONNX Conversion failed: Could not get cgImage from resized image")
            return nil
        }

        let width = cgImage.width
        let height = cgImage.height
        
        print("🔍 ONNX Input: CGImage dimensions: \(width)x\(height)")
        
        if width != 64 || height != 64 {
            print("❌ ONNX Conversion Error: Resized image is not 64x64. Actual: \(width)x\(height)")
            
            // 추가 디버그 정보
            print("🔍 UIImage size: \(resizedImage.size)")
            print("🔍 UIImage scale: \(resizedImage.scale)")
            print("🔍 Expected CGImage size: 64x64")
            print("🔍 Actual CGImage size: \(width)x\(height)")
            
            return nil
        }

        let colorSpace = CGColorSpaceCreateDeviceRGB()
        let bytesPerPixel = 4
        let bytesPerRow = bytesPerPixel * width
        let bitsPerComponent = 8
        
        var pixelData = [UInt8](repeating: 0, count: width * height * bytesPerPixel)
        
        guard let context = CGContext(
            data: &pixelData,
            width: width,
            height: height,
            bitsPerComponent: bitsPerComponent,
            bytesPerRow: bytesPerRow,
            space: colorSpace,
            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue | CGBitmapInfo.byteOrder32Big.rawValue
        ) else {
            print("❌ ONNX Conversion failed: Could not create CGContext.")
            return nil
        }
        
        context.draw(cgImage, in: CGRect(x: 0, y: 0, width: width, height: height))
        
        var tensorData = [Float]()
        tensorData.reserveCapacity(width * height * 3)
        
        // BGR 순서 유지 (모델이 BGR을 기대함)
        for i in 0..<(width * height) {
            let offset = i * bytesPerPixel
            let r = Float(pixelData[offset])
            let g = Float(pixelData[offset + 1])
            let b = Float(pixelData[offset + 2])
            
            tensorData.append(b) // Blue
            tensorData.append(g) // Green
            tensorData.append(r) // Red
        }
        
        let result = tensorData.withUnsafeBufferPointer { Data(buffer: $0) }
        print("✅ ONNX Input: Generated BGR tensor data of size: \(result.count) bytes")
        
        return result
    }
  
   

  // MARK: - Detection Methods
  func detect(image: UIImage) -> FaceLandmarkDetectionResultBundle? {
    guard let mpImage = try? MPImage(uiImage: image) else {
      return nil
    }
    do {
      let startDate = Date()
      let result = try faceLandmarker?.detect(image: mpImage)
      let inferenceTime = Date().timeIntervalSince(startDate) * 1000
      
      var faceFeatures: FaceFeatures? = nil
      if let result = result, !result.faceLandmarks.isEmpty {
        faceFeatures = calculateFaceFeatures(
          landmarks: result.faceLandmarks[0],
          imageWidth: Int(image.size.width),
          imageHeight: Int(image.size.height)
        )
      }
      
      return FaceLandmarkDetectionResultBundle(
        inferenceTime: inferenceTime,
        faceLandmarkDetectorResults: [result],
        size: image.size,
        croppedFrame: nil,
        onnxInputData: nil,
        faceFeatures: faceFeatures
      )
    } catch {
      print(error)
      return FaceLandmarkDetectionResultBundle(
        inferenceTime: 0,
        faceLandmarkDetectorResults: [nil],
        size: image.size,
        croppedFrame: nil,
        onnxInputData: nil,
        faceFeatures: nil
      )
    }
  }

  func detectAsync(
    sampleBuffer: CMSampleBuffer,
    orientation: UIImage.Orientation,
    timeStamps: Int
  ) {
    guard let image = try? MPImage(sampleBuffer: sampleBuffer, orientation: orientation) else {
      return
    }
    
    self.currentUIImage = UIImage(sampleBuffer: sampleBuffer, orientation: orientation)
    
    do {
      self.livestreamImageSize = CGSize(width: image.width, height: image.height)
      try faceLandmarker?.detectAsync(image: image, timestampInMilliseconds: timeStamps)
    } catch {
      print(error)
    }
  }
}

// MARK: - FaceLandmarkDetectorLiveStreamDelegate
extension FaceLandmarkDetectorHelper: FaceLandmarkerLiveStreamDelegate {
  func faceLandmarker(
    _ faceLandmarker: FaceLandmarker,
    didFinishDetection result: FaceLandmarkerResult?,
    timestampInMilliseconds: Int,
    error: Error?
  ) {
    let inferenceTime = Date().timeIntervalSince1970 * 1000 - Double(timestampInMilliseconds)
    
    var croppedFrameData: Data? = nil
    var onnxInputData: Data? = nil
    var faceFeatures: FaceFeatures? = nil
    
    if let result = result, !result.faceLandmarks.isEmpty, let sourceImage = currentUIImage {
      faceFeatures = calculateFaceFeatures(
        landmarks: result.faceLandmarks[0],
        imageWidth: Int(livestreamImageSize.width),
        imageHeight: Int(livestreamImageSize.height)
      )
      
      var minX: Float = 1.0, minY: Float = 1.0, maxX: Float = 0.0, maxY: Float = 0.0
      result.faceLandmarks[0].forEach { landmark in
        minX = min(minX, landmark.x)
        minY = min(minY, landmark.y)
        maxX = max(maxX, landmark.x)
        maxY = max(maxY, landmark.y)
      }
      
      let margin: Float = 0.1
      let cropRect = CGRect(
        x: CGFloat(max(0.0, minX - margin)) * sourceImage.size.width,
        y: CGFloat(max(0.0, minY - margin)) * sourceImage.size.height,
        width: CGFloat(min(1.0, maxX + margin) - max(0.0, minX - margin)) * sourceImage.size.width,
        height: CGFloat(min(1.0, maxY + margin) - max(0.0, minY - margin)) * sourceImage.size.height
      )

      if cropRect.width > 0 && cropRect.height > 0, let croppedImage = cropImage(image: sourceImage, rect: cropRect) {
        
        // Resize to 192x192 ONCE and use for both JPEG and ONNX data generation.
        if let resizedImage192 = resizeImage(image: croppedImage, targetSize: CGSize(width: 192, height: 192)) {
            // Use the 192x192 image for JPEG data.
            croppedFrameData = resizedImage192.jpegData(compressionQuality: 0.8)
            
            // Pass the SAME 192x192 image to the ONNX converter.
            // It will be resized again to 64x64 internally, matching Android's flow.
            onnxInputData = convertImageToOnnxInput(image: resizedImage192)

            if onnxInputData == nil {
                print("❌ ONNX data generation returned nil.")
            }
        } else {
             print("❌ Failed to resize cropped image to 192x192.")
        }
      }
    }
    
    let resultBundle = FaceLandmarkDetectionResultBundle(
      inferenceTime: inferenceTime,
      faceLandmarkDetectorResults: [result],
      size: livestreamImageSize,
      croppedFrame: croppedFrameData,
      onnxInputData: onnxInputData,
      faceFeatures: faceFeatures
    )
    
    delegate?.faceLandmarkDetectorHelper(self, onResults: resultBundle, error: error)
  }
  
  private func cropImage(image: UIImage, rect: CGRect) -> UIImage? {
    guard let cgImage = image.cgImage?.cropping(to: rect) else {
      return nil
    }
    return UIImage(cgImage: cgImage, scale: image.scale, orientation: image.imageOrientation)
  }
}

/// A result from the `FaceLandmarkDetectorHelper`.
struct FaceLandmarkDetectionResultBundle {
  let inferenceTime: Double
  let faceLandmarkDetectorResults: [FaceLandmarkerResult?]
  let size: CGSize
  let croppedFrame: Data?
  let onnxInputData: Data?
  let faceFeatures: FaceFeatures?
}

// MARK: - UIImage extension for CMSampleBuffer
extension UIImage {
  convenience init?(sampleBuffer: CMSampleBuffer, orientation: UIImage.Orientation) {
    guard let imageBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { return nil }
    
    CVPixelBufferLockBaseAddress(imageBuffer, CVPixelBufferLockFlags(rawValue: 0))
    defer { CVPixelBufferUnlockBaseAddress(imageBuffer, CVPixelBufferLockFlags(rawValue: 0)) }
    
    let baseAddress = CVPixelBufferGetBaseAddress(imageBuffer)
    let bytesPerRow = CVPixelBufferGetBytesPerRow(imageBuffer)
    let width = CVPixelBufferGetWidth(imageBuffer)
    let height = CVPixelBufferGetHeight(imageBuffer)
    
    let colorSpace = CGColorSpaceCreateDeviceRGB()
    let bitmapInfo = CGBitmapInfo.byteOrder32Little.rawValue | CGImageAlphaInfo.premultipliedFirst.rawValue
    
    guard let context = CGContext(data: baseAddress, width: width, height: height,
                                  bitsPerComponent: 8, bytesPerRow: bytesPerRow,
                                  space: colorSpace, bitmapInfo: bitmapInfo),
          let cgImage = context.makeImage() else { return nil }
    
    self.init(cgImage: cgImage, scale: 1.0, orientation: orientation)
  }
}

