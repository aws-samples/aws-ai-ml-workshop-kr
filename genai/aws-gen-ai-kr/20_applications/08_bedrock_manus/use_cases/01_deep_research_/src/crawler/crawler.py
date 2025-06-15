# import sys

# from .article import Article
# from .jina_client import JinaClient
# from .readability_extractor import ReadabilityExtractor


# class Crawler:
#     def crawl(self, url: str) -> Article:
#         # To help LLMs better understand content, we extract clean
#         # articles from HTML, convert them to markdown, and split
#         # them into text and image blocks for one single and unified
#         # LLM message.
#         #
#         # Jina is not the best crawler on readability, however it's
#         # much easier and free to use.
#         #
#         # Instead of using Jina's own markdown converter, we'll use
#         # our own solution to get better readability results.
#         jina_client = JinaClient()
#         html = jina_client.crawl(url, return_format="html")
#         extractor = ReadabilityExtractor()
#         article = extractor.extract_article(html)
#         article.url = url
#         return article


# if __name__ == "__main__":
#     if len(sys.argv) == 2:
#         url = sys.argv[1]
#     else:
#         url = "https://fintel.io/zh-hant/s/br/nvdc34"
#     crawler = Crawler()
#     article = crawler.crawl(url)
#     print(article.to_markdown())


import sys
import requests
from bs4 import BeautifulSoup
from .article import Article

class Crawler:
    def crawl(self, url: str) -> Article:
        # 브라우저처럼 보이는 사용자 에이전트 설정
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Referer': 'https://www.google.com/',
            'DNT': '1',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
        }
        
        # requests를 사용하여 웹 페이지 가져오기
        response = requests.get(url, headers=headers)
        response.raise_for_status()  # 오류 발생 시 예외 발생
        
        # BeautifulSoup을 사용하여 HTML 파싱
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # 주요 콘텐츠 추출
        # 이 부분은 웹사이트 구조에 따라 조정 필요
        title = soup.title.string if soup.title else ""
        
        # 본문 추출 (간단한 예시, 웹사이트에 맞게 수정 필요)
        content = ""
        main_content = soup.find('main') or soup.find('article') or soup.find('div', class_='content')
        if main_content:
            paragraphs = main_content.find_all('p')
            content = '\n\n'.join([p.get_text() for p in paragraphs])
        
        # 이미지 추출
        images = []
        img_tags = soup.find_all('img')
        for img in img_tags:
            src = img.get('src')
            if src and src.startswith('http'):
                images.append(src)

        #print ("title", title)
        #print ("content", content)
        #print ("images", images)
        
        # Article 객체 생성 및 반환
        article = Article(
            title=title,
            html_content=content
        )
        #article.url = url
        #article.title = title
        #article.html_content = content
        #article.images = images
        
        return article

if __name__ == "__main__":
    if len(sys.argv) == 2:
        url = sys.argv[1]
    else:
        url = "https://fintel.io/zh-hant/s/br/nvdc34"
    crawler = Crawler()
    article = crawler.crawl(url)
    print(article.to_markdown())