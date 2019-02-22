// Include Directories
// ==========================================
#include "darknet.h"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>
#include <string>

char *voc_names[] = { "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor" };

cv::Mat image_to_mat(image im);
image mat_to_image(cv::Mat* src);

void drawDetections(network* net, int nDetections, detection* detections, cv::Mat& img, char** classNames)
{
	/*
	Bbox x & y positions are centered. 
	*/
	int width = img.cols;
	int height = img.rows;

	for (int i = 0; i < nDetections; i++)
	{
		detection* d = detections+i;
		cv::Rect2i bbox(d->bbox.x*width - d->bbox.w*width/2.0, d->bbox.y*height - d->bbox.h*height/2.0, d->bbox.w*width, d->bbox.h*height);
		cv::rectangle(img, bbox, {0,255,0},2);
		//cv::putText(img, classNames[d->classes], {int(d->bbox.x*width - d->bbox.w*width / 2.0), int(d->bbox.y*height - d->bbox.h*height / 2.0)}, 0, 2.0, {180,180,180},1,8,true);
	}
}

int main(int argc, char* argv[])
{
	// Load the program info from the command line text file
	char* cfgfile = NULL;//"C:/Builds/Official/DarkNet/cfg/yolov3.cfg";
	char* weightfile = NULL;//"C:/Builds/Official/DarkNet/weights/yolov3.weights";
	char* baseDirectory = NULL; //"C:/Builds/Official/DarkNet/";
	char* videoFile = NULL; 
	std::ifstream in(argv[1], std::ifstream::in);
	std::string line;
	if (in.is_open())
	{
		int count = 0;
		while (std::getline(in, line))
		{
			switch (count)
			{
			case 0:	// cfgfile
				cfgfile = static_cast<char*>(malloc(line.size()));
				strcpy(cfgfile, line.c_str());
				break;
			case 1:	// weightfile
				weightfile = static_cast<char*>(malloc(line.size()));
				strcpy(weightfile, line.c_str());
				break;
			case 2:	// baseDirectory
				baseDirectory = static_cast<char*>(malloc(line.size()));
				strcpy(baseDirectory, line.c_str());
				break;
			case 3:	// videoFile
				videoFile = static_cast<char*>(malloc(line.size()));
				strcpy(videoFile, line.c_str());
				break;
			}

			count++;
		}
	}
	

	// Yolo preamble
	image **alphabet = load_alphabet(baseDirectory);
	network *net = load_network(cfgfile, weightfile, 0);
	layer l = net->layers[net->n - 1];
	set_batch_network(net, 1);
	srand(2222222);
	clock_t time;
	char buff[256];
	char *input = buff;
	float nms = .4;
	float thresh = 0.5;

	// Load the video with openCV
	cv::VideoCapture cap(videoFile);
	cv::Mat img;
	cv::namedWindow("output", cv::WINDOW_KEEPRATIO);

	// Classify the video with YOLO
	while (cap.isOpened())
	{
		if (!cap.read(img)) { std::cout << "Bad frame, skipping\n"; continue; }
		image im = mat_to_image(&img);
		image sized = resize_image(im, net->w, net->h);
		float *X = sized.data;
		time = clock();
		network_predict(net, X);
		int nboxes = 0;
		detection *dets = get_network_boxes(net, 1, 1, thresh, 0, 0, 0, &nboxes);
		printf("Predicted %d bounding boxes in %f seconds.\n", nboxes, sec(clock() - time));
		if (nms) do_nms_sort(dets, l.side*l.side*l.n, l.classes, nms);

		//draw_detections(im, dets, l.side*l.side*l.n, thresh, voc_names, alphabet, 20);
		drawDetections(net, nboxes, dets, img, voc_names);

		//img = image_to_mat(im);
		cv::imshow("output", img);
		cv::waitKey(1);

		free_detections(dets, nboxes);
		free_image(im);
		free_image(sized);

	}

	// Free the allocated memory
	free(cfgfile);
	free(weightfile);
	free(baseDirectory);

	return 0;
}

