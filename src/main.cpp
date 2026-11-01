#include <opencv2/opencv.hpp>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <vector>

struct Light {
    cv::Point2f top;
    cv::Point2f bottom;
    cv::Point2f center;
};

int main() {
    std::string inputPath = "../data/input.avi";
    std::string outputPath = "out.avi";

    cv::VideoCapture capture(inputPath);

    if (!capture.isOpened()) {
        std::cout << "Cannot open input video." << std::endl;
        return 1;
    }

    double fps = capture.get(cv::CAP_PROP_FPS);
    if (!std::isfinite(fps) || fps <= 0) {
        fps = 30;
    }

    int delay = std::max(1, static_cast<int>(1000.0 / fps));

    cv::VideoWriter writer;
    cv::Mat frame;

    while (capture.read(frame)) {
        if (frame.empty()) {
            break;
        }

        cv::Mat gray, brightMask;
        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
        cv::threshold(gray, brightMask, 60, 255, cv::THRESH_BINARY);

        std::vector<cv::Mat> channels;
        cv::split(frame, channels);

        cv::Mat difference, redMask;
        cv::subtract(channels[2], channels[0], difference);
        cv::threshold(difference, redMask, 40, 255, cv::THRESH_BINARY);

        cv::Mat mask;
        cv::bitwise_and(brightMask, redMask, mask);

        cv::Mat kernel = cv::getStructuringElement(
            cv::MORPH_RECT, cv::Size(3, 3));
        cv::morphologyEx(mask, mask, cv::MORPH_CLOSE, kernel);

        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(
            mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

        std::vector<Light> lights;

        for (const auto& contour : contours) {
            if (cv::contourArea(contour) < 8) {
                continue;
            }

            cv::RotatedRect rect = cv::minAreaRect(contour);

            cv::Point2f points[4];
            rect.points(points);

            double edge1 = cv::norm(points[1] - points[0]);
            double edge2 = cv::norm(points[2] - points[1]);

            double length = std::max(edge1, edge2);
            double width = std::min(edge1, edge2);

            if (width < 1 || length < 6) {
                continue;
            }

            double ratio = length / width;

            if (ratio < 2 || ratio > 25) {
                continue;
            }

            cv::Point2f top, bottom;

            if (edge1 < edge2) {
                top = (points[0] + points[1]) * 0.5f;
                bottom = (points[2] + points[3]) * 0.5f;
            } else {
                top = (points[1] + points[2]) * 0.5f;
                bottom = (points[3] + points[0]) * 0.5f;
            }

            if (top.y > bottom.y) {
                std::swap(top, bottom);
            }

            float dx = std::abs(bottom.x - top.x);
            float dy = std::abs(bottom.y - top.y);

            if (dx > dy) {
                continue;
            }

            Light light;
            light.top = top;
            light.bottom = bottom;
            light.center = rect.center;

            lights.push_back(light);
        }

        if (lights.size() == 2) {
            if (lights[0].center.x > lights[1].center.x) {
                std::swap(lights[0], lights[1]);
            }

            cv::line(frame, lights[0].top, lights[1].top,
                     cv::Scalar(0, 255, 0), 2);

            cv::line(frame, lights[1].top, lights[1].bottom,
                     cv::Scalar(0, 255, 0), 2);

            cv::line(frame, lights[1].bottom, lights[0].bottom,
                     cv::Scalar(0, 255, 0), 2);

            cv::line(frame, lights[0].bottom, lights[0].top,
                     cv::Scalar(0, 255, 0), 2);
        }

        cv::Mat maskBgr, result;
        cv::cvtColor(mask, maskBgr, cv::COLOR_GRAY2BGR);
        cv::hconcat(frame, maskBgr, result);

        if (!writer.isOpened()) {
            writer.open(
                outputPath,
                cv::VideoWriter::fourcc('M', 'J', 'P', 'G'),
                fps,
                result.size()
            );

            if (!writer.isOpened()) {
                std::cout << "Cannot create output video." << std::endl;
                return 1;
            }
        }

        writer.write(result);
        cv::imshow("Result", result);

        if (cv::waitKey(delay) == 27) {
            break;
        }
    }

    capture.release();
    writer.release();
    cv::destroyAllWindows();

    return 0;
}
