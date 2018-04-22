#include "predict.h"

namespace nerd {

PredictionWindow::PredictionWindow(QWidget*)
    : channel_(grpc::CreateChannel("localhost:50051",
                                   grpc::InsecureChannelCredentials())),
      stub_(LeagueMatchPrediction::NewStub(channel_)) {
  setupUi(this);

  connect(cbBlueTop, SIGNAL(activated(int)),
          this, SLOT(OnSelectChange(int)));
  connect(cbBlueJungle, SIGNAL(activated(int)),
          this, SLOT(OnSelectChange(int)));
  connect(cbBlueMid, SIGNAL(activated(int)),
          this, SLOT(OnSelectChange(int)));
  connect(cbBlueSupport, SIGNAL(activated(int)),
          this, SLOT(OnSelectChange(int)));
  connect(cbBlueADC, SIGNAL(activated(int)),
          this, SLOT(OnSelectChange(int)));

  connect(cbRedTop, SIGNAL(activated(int)),
          this, SLOT(OnSelectChange(int)));
  connect(cbRedJungle, SIGNAL(activated(int)),
          this, SLOT(OnSelectChange(int)));
  connect(cbRedMid, SIGNAL(activated(int)),
          this, SLOT(OnSelectChange(int)));
  connect(cbRedSupport, SIGNAL(activated(int)),
          this, SLOT(OnSelectChange(int)));
  connect(cbRedADC, SIGNAL(activated(int)),
          this, SLOT(OnSelectChange(int)));

  GetChampions("league.sqlite3");

  grpc::ClientContext context;
  ServerStatusRequest request;
  ServerStatus server_status;
  grpc::Status stat = stub_->GetServerStatus(&context, request, &server_status);
  if (!stat.ok() || !server_status.ok()) {
    LOG(WARNING) << "server not ready";
  } else {
    LOG(INFO) << "server ready...";
  }
}

void PredictionWindow::OnSelectChange(int index) {
  LOG(INFO) << "on select";
  PredictMatch();
}

int PredictionWindow::SqliteCallback(void* data, int, char** columns, char**) {
  QMap<QString, int>* champs = reinterpret_cast<QMap<QString, int>*>(data);
  LOG(INFO) << columns[1] << " loaded.";
  champs->insert(columns[1] ? columns[1] : "", std::stoi(columns[0]));
  return 0;
}

bool PredictionWindow::GetChampions(const std::string& dbname) {
  sqlite3* db_handle;
  if (sqlite3_open(dbname.c_str(), &db_handle)) {
    LOG(FATAL) << "fail to open " << dbname;
    return false;
  }
  LOG(INFO) << dbname << " opened.";

  char* error_msg = nullptr;
  if (sqlite3_exec(db_handle, "SELECT id, name FROM champs;",
                   PredictionWindow::SqliteCallback, &champs_,
                   &error_msg) != SQLITE_OK) {
    LOG(FATAL) << error_msg;
    return false;
  }

  sqlite3_close(db_handle);
  QList<QString> champs = champs_.keys();
  qSort(champs.begin(), champs.end());

  cbBlueTop->addItems(champs);
  cbBlueJungle->addItems(champs);
  cbBlueMid->addItems(champs);
  cbBlueSupport->addItems(champs);
  cbBlueADC->addItems(champs);

  cbRedTop->addItems(champs);
  cbRedJungle->addItems(champs);
  cbRedMid->addItems(champs);
  cbRedSupport->addItems(champs);
  cbRedADC->addItems(champs);
  return true;
}

void PredictionWindow::PredictMatch() {
  grpc::ClientContext context;
  Match match;
  MatchPrediction pred;

  Team* blue = new Team();
  blue->set_top(champs_[cbBlueTop->currentText()]);
  blue->set_jungle(champs_[cbBlueJungle->currentText()]);
  blue->set_mid(champs_[cbBlueMid->currentText()]);
  blue->set_support(champs_[cbBlueSupport->currentText()]);
  blue->set_adc(champs_[cbBlueADC->currentText()]);
  match.set_allocated_blue(blue);

  Team* red = new Team();
  red->set_top(champs_[cbRedTop->currentText()]);
  red->set_jungle(champs_[cbRedJungle->currentText()]);
  red->set_mid(champs_[cbRedMid->currentText()]);
  red->set_support(champs_[cbRedSupport->currentText()]);
  red->set_adc(champs_[cbRedADC->currentText()]);
  match.set_allocated_blue(red);

  grpc::Status stat = stub_->Predict(&context, match, &pred);
  if (!stat.ok()) {
    LOG(WARNING) << "fail to predict match";
  }

  if (pred.winning_team() == "red") {
    QPalette palette = lbWinningTeam->palette();
    palette.setColor(lbWinningTeam->foregroundRole(), Qt::red);
    lbWinningTeam->setPalette(palette);
  } else if (pred.winning_team() == "blue") {
    QPalette palette = lbWinningTeam->palette();
    palette.setColor(lbWinningTeam->foregroundRole(), Qt::blue);
    lbWinningTeam->setPalette(palette);
  }

  lbWinningTeam->setText(QString(pred.winning_team().c_str()));
  lbProbability->setText(QString::number(pred.probability()));
}

}  // namespace nerd
