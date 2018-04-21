#include <QApplication>

#include "predict.h"

int main(int argc, char** argv) {
  google::InitGoogleLogging(argv[0]);
  google::LogToStderr();

  QApplication app(argc, argv);
  nerd::PredictionWindow window(nullptr);
  window.show();
  return app.exec();
}
