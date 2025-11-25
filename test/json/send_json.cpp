#include <iostream>
#include <string>
#include <curl/curl.h>
#include <chrono>
#include <iomanip>
#include <sstream>

static size_t WriteCallback(void* contents, size_t size, size_t nmemb, std::string* output) {
    output->append((char*)contents, size * nmemb);
    return size * nmemb;
}

int main() {
    // Generar timestamp local de la laptop
    auto now = std::chrono::system_clock::now();
    std::time_t t = std::chrono::system_clock::to_time_t(now);
    std::tm tm_local;

#ifdef _WIN32
    localtime_s(&tm_local, &t);
#else
    localtime_r(&t, &tm_local);
#endif

    std::stringstream ss;
    ss << std::put_time(&tm_local, "%Y-%m-%d %H:%M:%S");
    std::string timestamp = ss.str();

    CURL* curl = curl_easy_init();
    if (!curl) {
        std::cerr << "Could not init curl\n";
        return 1;
    }

    std::string url = "https://fn-va-panto.azurewebsites.net/api/camera-facial-recognition-freh";

    // JSON usando el timestamp generado
    std::string json_body = "{"
        "\"Enterprise\":\"freh\","
        "\"Id\":\"user001\","
        "\"Gender\":\"male\","
        "\"Timestamp\":\"" + timestamp + "\","
        "\"Age\":30,"
        "\"Emotion\":\"neutral\""
    "}";

    struct curl_slist* headers = nullptr;
    headers = curl_slist_append(headers, "Content-Type: application/json");

    curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
    curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
    curl_easy_setopt(curl, CURLOPT_POSTFIELDS, json_body.c_str());
    curl_easy_setopt(curl, CURLOPT_SSL_VERIFYPEER, 0L);
    curl_easy_setopt(curl, CURLOPT_SSL_VERIFYHOST, 0L);

    std::string response;
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteCallback);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &response);

    std::cout << json_body << std::endl;

    CURLcode res = curl_easy_perform(curl);
    long http_code = 0;
    curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &http_code);

    std::cout << "\nHTTP " << http_code << ": " << response << std::endl;

    curl_slist_free_all(headers);
    curl_easy_cleanup(curl);
    return 0;
}
