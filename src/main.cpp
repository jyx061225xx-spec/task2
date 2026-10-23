#include <opencv2/opencv.hpp>
#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <string>
#include <vector>

struct Params {
    int brightness = 60;
    int colorDifference = 40;
    double minArea = 8.0;
    double minLength = 6.0;
    double minAspect = 2.0;
    double maxAspect = 25.0;
    double maxTilt = 40.0;
    double maxLengthRatio = 1.5;
    double maxAngleDifference = 15.0;
    double maxYRatio = 0.6;
    double minSpacingRatio = 0.7;
    double maxSpacingRatio = 5.0;
};

struct Light {
    cv::Point2f top, bottom, center;
    double length, angle;
};
struct PairCandidate {
    int left, right;
    double score;
};

cv::Mat makeMask(const cv::Mat& frame, const std::string& color,
                 const Params& p) {
    cv::Mat gray, brightMask;
    cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
    cv::threshold(gray, brightMask, p.brightness, 255, cv::THRESH_BINARY);
    cv::Mat mask;
    if (color == "any") {
        mask = brightMask.clone();
    } else {
        std::vector<cv::Mat> channels;
        cv::split(frame, channels); 
        cv::Mat difference, colorMask;
        if (color == "blue") {
            cv::subtract(channels[0], channels[2], difference);
        } else {
            cv::subtract(channels[2], channels[0], difference);
        }
        cv::threshold(difference, colorMask, p.colorDifference,
                      255, cv::THRESH_BINARY);
        cv::bitwise_and(brightMask, colorMask, mask);
    }
    
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
    cv::morphologyEx(mask, mask, cv::MORPH_CLOSE, kernel);
    return mask;
}

std::vector<Light> findLights(const cv::Mat& mask, const Params& p) {
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    std::vector<Light> lights;
    for (const auto& contour : contours) {
        if (cv::contourArea(contour) < p.minArea) continue;
        cv::RotatedRect rect = cv::minAreaRect(contour);
        cv::Point2f v[4];
        rect.points(v);
        double edge01 = cv::norm(v[1] - v[0]);
        double edge12 = cv::norm(v[2] - v[1]);
        double length = std::max(edge01, edge12);
        double width = std::min(edge01, edge12);
        if (width < 1.0 || length < p.minLength) continue;
        double aspect = length / width;
        if (aspect < p.minAspect || aspect > p.maxAspect) continue;

        cv::Point2f a, b;

        if (edge01 < edge12) {
            a = (v[0] + v[1]) * 0.5f;
            b = (v[2] + v[3]) * 0.5f;
        } else {
            a = (v[1] + v[2]) * 0.5f;
            b = (v[3] + v[0]) * 0.5f;
        }
        if (a.y > b.y) std::swap(a, b);
        cv::Point2f direction = b - a;
        double angle = std::atan2(direction.x, direction.y) * 180.0 / CV_PI;
        if (std::abs(angle) > p.maxTilt) continue;
        lights.push_back({a, b, rect.center, length, angle});
    }
    std::sort(lights.begin(), lights.end(), [](const Light& a, const Light& b) {
        return a.center.x < b.center.x;
    });
    return lights;
}

std::vector<PairCandidate> matchLights(const std::vector<Light>& lights,
                                     const Params& p) {
    std::vector<PairCandidate> candidates;
    for (int i = 0; i < static_cast<int>(lights.size()); ++i) {
        for (int j = i + 1; j < static_cast<int>(lights.size()); ++j) {
            const Light& left = lights[i];
            const Light& right = lights[j];
            double meanLength = (left.length + right.length) / 2.0;
            double lengthRatio = std::max(left.length, right.length) /
                                 std::min(left.length, right.length);
            double angleDifference = std::abs(left.angle - right.angle);
            double yRatio = std::abs(left.center.y - right.center.y) / meanLength;
            double spacingRatio = (right.center.x - left.center.x) / meanLength;
            if (lengthRatio > p.maxLengthRatio ||
                angleDifference > p.maxAngleDifference ||
                yRatio > p.maxYRatio ||
                spacingRatio < p.minSpacingRatio ||
                spacingRatio > p.maxSpacingRatio) continue;
            double score = (lengthRatio - 1.0) +
                           angleDifference / 15.0 + yRatio;
            candidates.push_back({i, j, score});
        }
    }
    std::sort(candidates.begin(), candidates.end(),
              [](const PairCandidate& a, const PairCandidate& b) {
                  return a.score < b.score;
              });
    std::vector<bool> used(lights.size(), false);
    std::vector<PairCandidate> pairs;
    for (const auto& candidate : candidates) {
        if (used[candidate.left] || used[candidate.right]) continue;
        pairs.push_back(candidate);
        used[candidate.left] = used[candidate.right] = true;
    }
    return pairs;
}

void drawResult(cv::Mat& result, const std::vector<Light>& lights,
                const std::vector<PairCandidate>& pairs) {
    for (const auto& light : lights) {
        cv::line(result, light.top, light.bottom, cv::Scalar(0, 255, 255), 2);
        cv::circle(result, light.top, 3, cv::Scalar(0, 0, 255), -1);
        cv::circle(result, light.bottom, 3, cv::Scalar(255, 0, 0), -1);
    }
    for (const auto& pair : pairs) {
        const Light& left = lights[pair.left];
        const Light& right = lights[pair.right];
        std::array<cv::Point2f, 4> corners = {
            left.top, right.top, right.bottom, left.bottom
        };
        for (int k = 0; k < 4; ++k) {
            cv::line(result, corners[k], corners[(k + 1) % 4],
                     cv::Scalar(0, 255, 0), 2);
            cv::putText(result, std::to_string(k), corners[k],
                        cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 1);
        }
    }
}

int main(int argc, char** argv) {
    if (argc < 4 || argc > 5) {
        std::cerr << "Usage: " << argv[0]
                  << " input_video output.avi blue|red|any [--no-gui]\n";
        return 1;
    }
    std::string color = argv[3];
    if (color != "blue" && color != "red" && color != "any") {
        std::cerr << "Color must be blue, red or any.\n";
        return 1;
    }
    if (argc == 5 && std::string(argv[4]) != "--no-gui") {
        std::cerr << "Unknown option; expected --no-gui.\n";
        return 1;
    }
    bool show = argc == 4;
    try {
        Params p;
        cv::VideoCapture capture(argv[1]);
        if (!capture.isOpened()) {
            std::cerr << "Cannot open input video: " << argv[1] << '\n';
            return 1;
        }
        double fps = capture.get(cv::CAP_PROP_FPS);
        if (!std::isfinite(fps) || fps <= 0.0) fps = 30.0;
        int delay = std::max(1, static_cast<int>(std::round(1000.0 / fps)));
        cv::VideoWriter writer;
        cv::Mat frame;
        int frameIndex = 0;
        bool paused = false;
        while (capture.read(frame)) {
            if (frame.empty()) break;
            cv::Mat mask = makeMask(frame, color, p);
            auto lights = findLights(mask, p);
            auto pairs = matchLights(lights, p);
            cv::Mat result = frame.clone();
            drawResult(result, lights, pairs);
            ++frameIndex;
            std::string info = "Frame " + std::to_string(frameIndex) +
                "  Lights " + std::to_string(lights.size()) +
                "  Armors " + std::to_string(pairs.size());
            cv::putText(result, info, cv::Point(10, 25),
                        cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 0), 1);
            cv::Mat maskBgr, combined;
            cv::cvtColor(mask, maskBgr, cv::COLOR_GRAY2BGR);
            cv::hconcat(result, maskBgr, combined);
            if (!writer.isOpened()) {
                writer.open(argv[2], cv::VideoWriter::fourcc('M', 'J', 'P', 'G'),
                            fps, combined.size(), true);
                if (!writer.isOpened()) {
                    std::cerr << "Cannot create output. Check directory and MJPG support.\n";
                    return 1;
                }
            }
            writer.write(combined);
            if (frameIndex % 100 == 0) std::cout << info << '\n';
            if (show) {
                cv::imshow("Result (left) | Mask (right)", combined);
                bool nextFrame = false;
                while (!nextFrame) {
                    int key = cv::waitKey(paused ? 0 : delay) & 255;
                    if (key == 'q' || key == 27) {
                        std::cout << "Stopped early at frame " << frameIndex << '\n';
                        return 0;
                    }
                    if (key == ' ') {
                        paused = !paused;
                        nextFrame = !paused;
                    } else if (key == 'n') {
                        paused = true;
                        nextFrame = true;
                    } else if (!paused) {
                        nextFrame = true;
                    }
                }
            }
        }
        if (frameIndex == 0) {
            std::cerr << "No frames decoded. Check the video and decoder support.\n";
            return 1;
        }
        writer.release();
        capture.release();
        if (show) cv::destroyAllWindows();
        std::cout << "Read ended. Processed " << frameIndex
                  << " frames. Output: " << argv[2] << '\n';
        return 0;
    } catch (const cv::Exception& e) {
        std::cerr << "OpenCV error: " << e.what()
                  << "\nIf this is a display error, retry with --no-gui.\n";
        return 1;
    }
}
