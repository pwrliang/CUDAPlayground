#include <gflags/gflags.h>
#include <glog/logging.h>

#include "main.h"

int main(int argc, char *argv[]) {
  FLAGS_stderrthreshold = 0;

  gflags::ParseCommandLineFlags(&argc, &argv, true);
  gflags::ShutDownCommandLineFlags();

  google::InitGoogleLogging(argv[0]);
  google::InstallFailureSignalHandler();

  Run();

  google::ShutdownGoogleLogging();
}
