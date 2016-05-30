#ifndef __LLC_CODING__
#define __LLC_CODING__
using namespace cv;

static void LLC_Coding(const cv::Mat &feavec,
		const cv::Mat &codebook, const LLC_para* LLCpara,
		cv::Mat &Coeff);
		
struct LLC_para
{
	unsigned int KNN_Build_Num_Of_Tree=4;// construct an randomized kd-tree index using 4 kd-trees
	unsigned int KNN_traverse_time=16;	// larger means more accurate but costs more time
	unsigned int knn_k=5;			// number of nearest neighbour to find, 5 is optimal according to paper
	unsigned int Num_of_pyramid_level=3;	// 3 means [1x1, 2x2, 4x4]
};		
#endif
