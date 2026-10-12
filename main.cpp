// Belief Propagation for Stereo Matching
// GitHub Repository: https://github.com/bollasap/belief-propagation-stereo
//

#include <iostream>
#include <string>
#include <vector>
#include <chrono>
#include <algorithm>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

typedef vector<int> IntRow;
typedef vector<IntRow> IntTable;
typedef vector<int> Msg;
typedef vector<Msg> MsgRow;
typedef vector<MsgRow> MsgTable;

class BP
{
private:
	int width, height, nextLine, bufferSize;
	IntTable smoothnessCost;
	Mat leftImage, rightImage, dispMap, dispImage;
	MsgTable dataCost, msgUp, msgDown, msgRight, msgLeft;
	MsgTable msgUp2, msgDown2, msgRight2, msgLeft2;

	void lowMemoryImplementation();
	void updateBuffer(int n);
	void standardImplementation();
	void updateMessages();
	void sendMessageUp(int x, int y, bool secondary = false);
	void sendMessageDown(int x, int y, bool secondary = false);
	void sendMessageRight(int x, int y, bool secondary = false);
	void sendMessageLeft(int x, int y, bool secondary = false);
	void createMessage(Msg &msgData, Msg &msgIn1, Msg &msgIn2, Msg &msgIn3, Msg &msgOut);
	double calcDataCost(int x, int y, int label);
	int calcSmoothnessCost(int label1, int label2);
	int findBestAssignment(int x, int y);
	int calcBelief(int x, int y, int label);
	int calcEnergy();
	void showDisparityImage();

public:
	bool sgmLike, lowMemory, colorMap;
	int dispLevels, updateSchedule, iterations, windowSize;
	int smoothnessWeight, smoothnessThreshold, sgmP1, sgmP2;
	int previousLabelsCost;

	Mat computeDisparity(Mat left, Mat right);
};

BP bp;
string inputFolder;
bool gaussianFilter, motion;

void runStereoMatching(string leftFile, string rightFile, string disparityFile);

// Start the program
int main()
{
	// Load the file containing the user's options
	FileStorage fs("Options.yml", ACCESS_READ);
	if (!fs.isOpened())
	{
		cerr << "Options file failed to open." << endl;
		exit(1);
	}

	// Retrieve the parameter values from the file
	inputFolder = (string)fs["Input Folder"];
	bp.dispLevels = (int)fs["Disparity Levels"];
	bp.updateSchedule = (int)fs["Update Schedule"];
	bp.iterations = (int)fs["Iterations"];
	gaussianFilter = (int)fs["Gaussian Filter"];
	bp.windowSize = (int)fs["Window Size"];
	bp.sgmLike = (int)fs["SGM Like"];
	bp.smoothnessWeight = (int)fs["Smoothness Weight"];
	bp.smoothnessThreshold = (int)fs["Smoothness Threshold"];
	bp.sgmP1 = (int)fs["SGM P1"];
	bp.sgmP2 = (int)fs["SGM P2"];
	bp.lowMemory = (int)fs["Low Memory"];
	bp.colorMap = (int)fs["Color Map"];
	motion = (int)fs["Motion"];
	bp.previousLabelsCost = (int)fs["Previous Labels Cost"];

	// Close the file
	fs.release();

	// Display the parameter values
	cout << "Input Folder: " << inputFolder << endl;
	cout << "Disparity Levels: " << bp.dispLevels << endl;
	cout << "Update Schedule: " << bp.updateSchedule << endl;
	cout << "Iterations: " << bp.iterations << endl;
	cout << "Gaussian Filter: " << gaussianFilter << endl;
	cout << "Window Size: " << bp.windowSize << endl;
	cout << "SGM Like: " << bp.sgmLike << endl;
	cout << "Smoothness Weight: " << bp.smoothnessWeight << endl;
	cout << "Smoothness Threshold: " << bp.smoothnessThreshold << endl;
	cout << "SGM P1: " << bp.sgmP1 << endl;
	cout << "SGM P2: " << bp.sgmP2 << endl;
	cout << "Low Memory: " << bp.lowMemory << endl;
	cout << "Color Map: " << bp.colorMap << endl;
	cout << "Motion: " << motion << endl;
	cout << "Previous Labels Cost: " << bp.previousLabelsCost << endl;
	cout << endl;

	// Start the timer
	auto start = chrono::steady_clock::now();

	if (motion)
	{
		// If the user selected it, perform stereo matching with motion (five image pairs)
		for (int i = 1; i <= 5; i++)
		{ 
			string leftFile = inputFolder + "Motion/Left_" + to_string(i) + ".png";
			string rightFile = inputFolder + "Motion/Right_" + to_string(i) + ".png";
			string disparityFile = "Disparity_" + to_string(i) + ".png";
			runStereoMatching(leftFile, rightFile, disparityFile);
		}
	}
	else
	{
		// Otherwise, perform stereo matching without motion (one image pair)
		string leftFile = inputFolder + "Left.png";
		string rightFile = inputFolder + "Right.png";
		string disparityFile = "Disparity.png";
		runStereoMatching(leftFile, rightFile, disparityFile);
	}
	
	// Stop the timer and display the time
	auto end = chrono::steady_clock::now();
	auto diff = end - start;
	cout << "Running Time: " << chrono::duration<double, milli>(diff).count() << " ms" << endl;
	cout << endl;

	waitKey(0);
	
	return 0;
}

// Perform the stereo matching process
void runStereoMatching(string leftFile, string rightFile, string disparityFile)
{
	// Read the images and convert them to grayscale images
	Mat left = imread(leftFile, IMREAD_GRAYSCALE);
	Mat right = imread(rightFile, IMREAD_GRAYSCALE);
	if (left.data == NULL || right.data == NULL)
	{
		cerr << "Left or right image failed to open." << endl;
		exit(2);
	}

	// Check if the images are the same size
	if (left.cols != right.cols || left.rows != right.rows)
	{
		cerr << "Sizes do not match." << endl;
		exit(3);
	}
	
	// If the user selected it, apply a Gaussian filter to the images
	if (gaussianFilter)
	{
		GaussianBlur(left, left, Size(5, 5), 0.68);
		GaussianBlur(right, right, Size(5, 5), 0.68);
	}

	// Create the disparity map
	Mat disp = bp.computeDisparity(left, right);

	// Save the final disparity map to a file
	bool flag = imwrite(disparityFile, disp);
	if (flag == false)
	{
		cerr << "Save image failed." << endl;
		exit(4);
	}
}

// Create the disparity map
Mat BP::computeDisparity(Mat left, Mat right)
{
	// Get the two images
	leftImage = left;
	rightImage = right;

	// Get the dimensions of the left image
	width = leftImage.cols;
	height = leftImage.rows;

	// Set the disparity map to the same dimensions as the left image
	if (dispMap.cols != width || dispMap.rows != height)
		dispMap = Mat::zeros(height, width, CV_8U);

	// Calculate and keep in memory all smoothness cost values
	smoothnessCost = IntTable(dispLevels, IntRow(dispLevels));
	for (int i = 0; i < dispLevels; i++)
		for (int j = 0; j < dispLevels; j++)
			smoothnessCost[i][j] = calcSmoothnessCost(i, j);

	// Use the mode selected by the user (standard / low-memory)
	if (lowMemory && updateSchedule >= 5 && updateSchedule <= 6)
		lowMemoryImplementation();
	else if (!lowMemory && updateSchedule >= 1 && updateSchedule <= 6)
		standardImplementation();
	
	// Display the disparity map
	showDisparityImage();

	return dispImage;
}

// Low-Memory mode
void BP::lowMemoryImplementation()
{
	int step;
	
	// Set initial values based on the message update schedule selected by the user
	switch (updateSchedule)
	{
	case 5:
		step = 1;
		bufferSize = iterations + 2;
		break;
	case 6:
		step = 2;
		bufferSize = iterations * 2 + 2;
		break;
	}

	// Initialize the buffer
	dataCost = MsgTable(bufferSize, MsgRow(width, Msg(dispLevels, 0)));
	msgUp = MsgTable(bufferSize, MsgRow(width, Msg(dispLevels, 0)));
	msgDown = MsgTable(bufferSize, MsgRow(width, Msg(dispLevels, 0)));
	msgRight = MsgTable(bufferSize, MsgRow(width, Msg(dispLevels, 0)));
	msgLeft = MsgTable(bufferSize, MsgRow(width, Msg(dispLevels, 0)));

	// Add the data for the first line of the image to the end of the buffer
	nextLine = 0;
	updateBuffer(1);
	int firstLine = nextLine - bufferSize;

	// Start processing
	while (firstLine < height - 2)
	{
		// Shift the buffer data upward and append the new data to the end
		updateBuffer(step);
		firstLine = nextLine - bufferSize;
		
		// For every line in the buffer except the last and the first, in order from bottom to top
		for (int y = bufferSize - 2; y >= 1; y--)
		{
			// Scan the line from left to right
			for (int x = 0; x < width; x++)
			{
				if (x < width - 1)
					sendMessageRight(x, y);
				if (firstLine + y < height - 1)
					sendMessageDown(x, y);
			}

			// Scan the line from right to left
			for (int x = width - 1; x >= 0; x--)
			{
				if (x > 0)
					sendMessageLeft(x, y);
				if (firstLine + y > 0)
					sendMessageUp(x, y);
			}
		}

		// Update the disparity map
		for (int y = 0; y < bufferSize; y++)
			for (int x = 0; x < width; x++)
			{
				int label = findBestAssignment(x, y);
				if (firstLine + y >= 0 && firstLine + y < height)
					dispMap.at<uchar>(firstLine + y, x) = label;
			}

		// Display the disparity map
		showDisparityImage();
	}
	
	// Calculate and display the energy
	int energy = calcEnergy();
	cout << "Energy: " << energy << endl;
}

// Shift the buffer data up by n lines and append the new data to the end
void BP::updateBuffer(int n)
{
	nextLine += n;
	int firstLine = nextLine - bufferSize;
	
	// Shift the data up
	for (int y = 0; y < bufferSize - n; y++)
		for (int x = 0; x < width; x++)
			for (int i = 0; i < dispLevels; i++)
			{
				dataCost[y][x][i] = dataCost[y + n][x][i];
				msgUp[y][x][i] = msgUp[y + n][x][i];
				msgDown[y][x][i] = msgDown[y + n][x][i];
				msgRight[y][x][i] = msgRight[y + n][x][i];
				msgLeft[y][x][i] = msgLeft[y + n][x][i];
			}
	
	// Append the new data to the last lines
	for (int y = bufferSize - n; y < bufferSize; y++)
		for (int x = 0; x < width; x++)
		{
			for (int i = 0; i < dispLevels; i++)
			{
				dataCost[y][x][i] = (int)calcDataCost(x, firstLine + y, i);
				msgUp[y][x][i] = 0;
				msgDown[y][x][i] = 0;
				msgRight[y][x][i] = 0;
				msgLeft[y][x][i] = 0;
			}
			
			// Assign a lower cost to the labels used in the previous disparity map
			int label = (firstLine + y < height) ? dispMap.at<uchar>(firstLine + y, x) : 0;
			if (label != 0)
			{
				msgUp[y][x][label] = previousLabelsCost;
				msgDown[y][x][label] = previousLabelsCost;
				msgRight[y][x][label] = previousLabelsCost;
				msgLeft[y][x][label] = previousLabelsCost;
			}
		}
}

// Standard mode
void BP::standardImplementation()
{
	// Calculate and keep in memory all data cost values
	dataCost = MsgTable(height, MsgRow(width, Msg(dispLevels)));
	for (int y = 0; y < height; y++)
		for (int x = 0; x < width; x++)
			for (int i = 0; i < dispLevels; i++)
				dataCost[y][x][i] = (int)calcDataCost(x, y, i);

	// Initialize the message arrays to zero
	msgUp = MsgTable(height, MsgRow(width, Msg(dispLevels, 0)));
	msgDown = MsgTable(height, MsgRow(width, Msg(dispLevels, 0)));
	msgRight = MsgTable(height, MsgRow(width, Msg(dispLevels, 0)));
	msgLeft = MsgTable(height, MsgRow(width, Msg(dispLevels, 0)));

	// Assign a lower cost to the labels used in the previous disparity map
	for (int y = 0; y < height; y++)
		for (int x = 0; x < width; x++)
		{
			int label = dispMap.at<uchar>(y, x);
			if (label != 0)
			{
				msgUp[y][x][label] = previousLabelsCost;
				msgDown[y][x][label] = previousLabelsCost;
				msgRight[y][x][label] = previousLabelsCost;
				msgLeft[y][x][label] = previousLabelsCost;
			}
		}

	// If the Synchronous message update schedule was selected, initialize the secondary arrays as well
	if (updateSchedule == 1)
	{
		msgUp2 = MsgTable(height, MsgRow(width, Msg(dispLevels, 0)));
		msgDown2 = MsgTable(height, MsgRow(width, Msg(dispLevels, 0)));
		msgRight2 = MsgTable(height, MsgRow(width, Msg(dispLevels, 0)));
		msgLeft2 = MsgTable(height, MsgRow(width, Msg(dispLevels, 0)));
	}

	cout << "Iter\tEnergy" << endl;
	cout << "----------------" << endl;

	// Start the loops
	for (int iter = 1; iter <= iterations; iter++)
	{
		// Update the messages
		updateMessages();

		// Update the disparity map
		for (int y = 0; y < height; y++)
			for (int x = 0; x < width; x++)
			{
				int label = findBestAssignment(x, y);
				dispMap.at<uchar>(y, x) = label;
			}

		// Calculate and display the energy
		int energy = calcEnergy();
		cout << iter << "\t" << energy << endl;

		// Display the disparity map
		showDisparityImage();
	}

	cout << endl;
}

// Updates the messages using the message update schedule selected by the user
void BP::updateMessages()
{
	switch (updateSchedule)
	{
	case 1:
		// Synchronous
		for (int y = 0; y < height; y++)
			for (int x = 0; x < width; x++)
			{
				if (x < width - 1)
					sendMessageRight(x, y, true);
				if (x > 0)
					sendMessageLeft(x, y, true);
				if (y < height - 1)
					sendMessageDown(x, y, true);
				if (y > 0)
					sendMessageUp(x, y, true);
			}
		msgLeft = msgLeft2;
		msgRight = msgRight2;
		msgUp = msgUp2;
		msgDown = msgDown2;
		break;

	case 2:
		// Bipartite
		for (int i = 0; i < 2; i++)
			for (int y = 0; y < height; y++)
				for (int x = (y + i) % 2; x < width; x += 2)
				{
					if (y > 0)
						sendMessageUp(x, y);
					if (y < height - 1)
						sendMessageDown(x, y);
					if (x < width - 1)
						sendMessageRight(x, y);
					if (x > 0)
						sendMessageLeft(x, y);
				}
		break;
	
	case 3:
		// BP-M (Accelerated)
		for (int y = 0; y < height; y++)
			for (int x = 0; x < width - 1; x++)
				sendMessageRight(x, y);
		for (int y = 0; y < height; y++)
			for (int x = width - 1; x >= 1; x--)
				sendMessageLeft(x, y);
		for (int x = 0; x < width; x++)
			for (int y = 0; y < height - 1; y++)
				sendMessageDown(x, y);
		for (int x = 0; x < width; x++)
			for (int y = height - 1; y >= 1; y--)
				sendMessageUp(x, y);
		break;
	
	case 4:
		// BP-S
		for (int y = 0; y < height; y++)
			for (int x = 0; x < width; x++)
			{
				if (x < width - 1)
					sendMessageRight(x, y);
				if (y < height - 1)
					sendMessageDown(x, y);
			}
		for (int y = height - 1; y >= 0; y--)
			for (int x = width - 1; x >= 0; x--)
			{
				if (x > 0)
					sendMessageLeft(x, y);
				if (y > 0)
					sendMessageUp(x, y);
			}
		break;
		
	case 5:
		// ByRow1
		for (int y = 0; y < height; y++)
		{
			for (int x = 0; x < width; x++)
			{
				if (x < width - 1)
					sendMessageRight(x, y);
				if (y < height - 1)
					sendMessageDown(x, y);
			}
			for (int x = width - 1; x >= 0; x--)
			{
				if (x > 0)
					sendMessageLeft(x, y);
				if (y > 0)
					sendMessageUp(x, y);
			}
		}
		break;

	case 6:
		// ByRow2
		for (int i = 1; i >= 0; i--)
			for (int y = i; y < height; y += 2)
			{
				for (int x = 0; x < width; x++)
				{
					if (x < width - 1)
						sendMessageRight(x, y);
					if (y < height - 1)
						sendMessageDown(x, y);
				}
				for (int x = width - 1; x >= 0; x--)
				{
					if (x > 0)
						sendMessageLeft(x, y);
					if (y > 0)
						sendMessageUp(x, y);
				}
			}
		break;
	}
}

// Sends a message to the pixel above
void BP::sendMessageUp(int x, int y, bool secondary)
{
	Msg &msgOut = (secondary) ? msgDown2[y - 1][x] : msgDown[y - 1][x];
	createMessage(dataCost[y][x], msgDown[y][x], msgRight[y][x], msgLeft[y][x], msgOut);
}

// Sends a message to the pixel below
void BP::sendMessageDown(int x, int y, bool secondary)
{
	Msg &msgOut = (secondary) ? msgUp2[y + 1][x] : msgUp[y + 1][x];
	createMessage(dataCost[y][x], msgUp[y][x], msgRight[y][x], msgLeft[y][x], msgOut);
}

// Sends a message to the pixel to the right
void BP::sendMessageRight(int x, int y, bool secondary)
{
	Msg &msgOut = (secondary) ? msgLeft2[y][x + 1] : msgLeft[y][x + 1];
	createMessage(dataCost[y][x], msgUp[y][x], msgDown[y][x], msgLeft[y][x], msgOut);
}

// Sends a message to the pixel to the left
void BP::sendMessageLeft(int x, int y, bool secondary)
{
	Msg &msgOut = (secondary) ? msgRight2[y][x - 1] : msgRight[y][x - 1];
	createMessage(dataCost[y][x], msgUp[y][x], msgDown[y][x], msgRight[y][x], msgOut);
}

// Creates a new message based on the user's selection (standard bp / sgm-like)
void BP::createMessage(Msg &msgData, Msg &msgIn1, Msg &msgIn2, Msg &msgIn3, Msg &msgOut)
{
	if (sgmLike)
	{
		// Create the vector (L)
		vector<int> L(dispLevels);
		for (int i = 0; i < dispLevels; i++)
			L[i] = msgData[i] + msgIn1[i] + msgIn2[i] + msgIn3[i];
		
		// Find the smallest value in the vector (minL)
		int minL = INT_MAX;
		for (int i = 0; i < dispLevels; i++)
			if (L[i] < minL)
				minL = L[i];

		// Use the smallest value (minL) to create the new message
		for (int i = 0; i < dispLevels; i++)
		{
			int cost1 = (i > 0) ? L[i - 1] + sgmP1 : INT_MAX;
			int cost2 = L[i];
			int cost3 = (i < dispLevels - 1) ? L[i + 1] + sgmP1 : INT_MAX;
			int cost4 = minL + sgmP2;

			msgOut[i] = min({ cost1, cost2, cost3, cost4 }) - minL;
		}
	}
	else
	{
		// Create the new message
		for (int i = 0; i < dispLevels; i++)
		{
			int min = INT_MAX;
			for (int j = 0; j < dispLevels; j++)
			{
				int cost = msgData[j] + smoothnessCost[i][j]
					+ msgIn1[j] + msgIn2[j] + msgIn3[j];
				if (cost < min)
					min = cost;
			}
			msgOut[i] = min;
		}

		// Normalize the message
		int min = INT_MAX;
		for (int i = 0; i < dispLevels; i++)
			if (msgOut[i] < min)
				min = msgOut[i];
		for (int i = 0; i < dispLevels; i++)
			msgOut[i] -= min;
	}
}

// Calculates the data cost using the window selected by the user
double BP::calcDataCost(int x, int y, int label)
{
	int sum = 0;
	for (int dy = -windowSize / 2; dy < windowSize / 2 + windowSize % 2; dy++)
		for (int dx = -windowSize / 2; dx < windowSize / 2 + windowSize % 2; dx++)
		{
			int leftPixel = (y + dy >= 0 && y + dy < height && x + dx >= 0 
				&& x + dx < width) ? leftImage.at<uchar>(y + dy, x + dx) : 0;
			int rightPixel = (y + dy >= 0 && y + dy < height && x + dx - label >= 0 
				&& x + dx - label < width) ? rightImage.at<uchar>(y + dy, x + dx - label) : 0;
			sum += abs(leftPixel - rightPixel);
		}
	double n = (windowSize * windowSize);
	double cost = sum / n;

	return cost;
}

// Calculates the smoothness cost
int BP::calcSmoothnessCost(int label1, int label2)
{
	int absoluteDiff = abs(label1 - label2);

	int cost;
	if (sgmLike)
		cost = (absoluteDiff == 0) ? 0 : (absoluteDiff == 1) ? sgmP1 : sgmP2;
	else
		cost = smoothnessWeight * min(absoluteDiff, smoothnessThreshold);

	return cost;
}

// Finds the best label for the pixel
int BP::findBestAssignment(int x, int y)
{
	int label, min = INT_MAX;
	for (int i = 0; i < dispLevels; i++)
	{
		int cost = calcBelief(x, y, i);
		if (cost < min)
		{
			label = i;
			min = cost;
		}
	}

	return label;
}

// Calculates the belief for a specific label
int BP::calcBelief(int x, int y, int label)
{
	int cost = dataCost[y][x][label]
		+ msgUp[y][x][label] + msgDown[y][x][label]
		+ msgRight[y][x][label] + msgLeft[y][x][label];

	return cost;
}

// Calculates the energy
int BP::calcEnergy()
{
	double energy = 0;
	for (int y = 0; y < height; y++)
		for (int x = 0; x < width; x++)
		{
			int label1 = dispMap.at<uchar>(y, x);
			energy += calcDataCost(x, y, label1);
			if (x < width - 1)
			{
				int label2 = dispMap.at<uchar>(y, x + 1);
				energy += smoothnessCost[label1][label2];
			}
			if (y < height - 1)
			{
				int label3 = dispMap.at<uchar>(y + 1, x);
				energy += smoothnessCost[label1][label3];
			}
		}

	return (int)energy;
}

// Displays the disparity map as an image
void BP::showDisparityImage()
{
	// Convert the disparity map to an image
	int scaleFactor = 256 / dispLevels;
	dispMap.convertTo(dispImage, CV_8U, scaleFactor);
	if (colorMap)
		applyColorMap(dispImage, dispImage, COLORMAP_JET);

	// Display the disparity map
	namedWindow("Disparity Image", WINDOW_NORMAL);
	imshow("Disparity Image", dispImage);
	waitKey(1);
}
