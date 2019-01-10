#include "ssd_iplugin.h"

/*
below layers must implement plugin in TensorRT-4.0.1.6
enum class PluginType : int
{
  kFASTERRCNN = 0,  //FasterRCNN fused plugin (RPN + ROI pooling)
  kNORMALIZE = 1,  //Normalize plugin
  kPERMUTE = 2, //Permute plugin
  kPRIORBOX = 3,  //PriorBox plugin
  kSSDDETECTIONOUTPUT = 4,  //SSD DetectionOutput plugin
  kConcat = 5,  //Concat plugin
  kPRELU = 6, //YOLO PReLU plugin
  kYOLOREORG = 7, //YOLO Reorg plugin
  kYOLOREGION = 8,  //YOLO Region plugin
  kANCHORGENERATOR = 9 //SSD Grid Anchor Generator
};
*/
