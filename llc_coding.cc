#include "llc_coding.h"
#include <iostream>
#include <opencv2/opencv.hpp>

using namespace cv;

static const auto LLC_beta=1e-4; // a very small non-zero number for the constraint part of LLC analytic solution
static void LLC_Coding(const cv::Mat &feavec,
		const cv::Mat &codebook, const LLC_para* LLCpara,
		cv::Mat &Coeff)
{
	cv::Mat IDX, Dst;
	auto k=LLCpara->knn_k;
	
	try{
		// build an kdtree for the codebook first before knn search
		cv::flann::Index cbindex;
		printf("Bullding KDTree...\n");
		cbindex.build(codebook, flann::KDTreeIndexParams(LLCpara->KNN_Build_Num_Of_Tree));	
		cbindex.knnSearch(feavec, IDX, Dst, k, 
					  cv::flann::SearchParams(LLCpara->KNN_traverse_time));			  
	}
	catch (std::exception& e){	std::cerr << e.what() << std::endl;}

	auto II = cv::Mat::eye(k,k,CV_32FC1);
	for(int i=0; i<feavec.rows; i++)
	{
		// one row of feaArr_t (i.e. length 128) copy to 5 identical rows
		auto z = cv::repeat(feavec.row(i), k, 1);
		auto idx_row_i = IDX.row(i);
		auto p = idx_row_i.begin<int>();
		for(int j=0; j<k; j++)
			z.row(j) = codebook.row(p[j]) - z.row(j);

		auto C = z*z.t(); // local variance of (B-xi)
		C = C+ II*LLC_beta*trace(C).val[0]; // regularlization (K>D)
		cv::Mat w = C.inv()*(cv::Mat::ones(k,1,CV_32FC1));
		w = w/sum(w).val[0]; // normalized w to sum to 1

		// Mat Coeff(nSmp, dSize, CV_32FC1);
		// Write rows of coeff whose corresponding column of codebook
		// is selected by flann, and write the weighting w into it
		//cv::MatConstIterator_<float> wp = w.begin<float>();
		p = idx_row_i.begin<int>();
		for(int j=0; j<k;j++)
			Coeff.at<float>(i,p[j]) = w.at<float>(j,0); //*wp;
			
	}
}
