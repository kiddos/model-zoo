#ifndef CAMERA_H
#define CAMERA_H

#include <QCamera>
#include <QCameraImageCapture>
#include <QMainWindow>
#include <QScopedPointer>

namespace yolo {

class Camera {
  Q_OBJECT

 public:
  Camera();

 public slots:
  void StartCamera();
  void StopCamera();

  void UpdateCameraDevices(QAction* action);
  void UpdateCameraState(QCamera::State);

 private:
  QScopedPointer<QCamera> cameras_;
  QScopedPointer<QCameraImageCapture> image_capture_;

  QImageEncoderSettings image_settings_;
};

} /* end of yolo namespace */

#endif /* end of include guard: CAMERA_H */
