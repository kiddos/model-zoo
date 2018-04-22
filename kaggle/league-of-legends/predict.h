#ifndef PREDICT_H
#define PREDICT_H

#include <grpc/grpc.h>
#include <grpc++/client_context.h>
#include <grpc++/channel.h>
#include <grpc++/create_channel.h>
#include <grpc++/security/credentials.h>
#include <sqlite3.h>
#include <glog/logging.h>

#include "league.pb.h"
#include "league.grpc.pb.h"
#include "ui_predict.h"

namespace nerd {

class PredictionWindow : public QWidget, public Ui::PredictWidget {
  Q_OBJECT

 public:
  PredictionWindow(QWidget* parent);

 public slots:
  void OnSelectChange(int index);

 private:
  static int SqliteCallback(void* data, int i, char** columns,
                            char** columns_names);
  bool GetChampions(const std::string& dbname);
  void PredictMatch();

  std::shared_ptr<grpc::Channel> channel_;
  std::unique_ptr<LeagueMatchPrediction::Stub> stub_;
  QMap<QString, int> champs_;
};

}  // namespace nerd

#endif /* end of include guard: PREDICT_H */
