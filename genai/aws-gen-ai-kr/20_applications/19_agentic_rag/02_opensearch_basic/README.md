# OpenSearch 기본 연습

## 1. 설치

## 2. Dev tools
이 워크샵은 주로 파이썬 코드와 langchain을 이용하여 OpenSearch와 통신하게 됩니다. 하지만 그 전에 Lab2에서는 순수한 OpenSearch API만을 사용하여 기본적인 이해를 돕습니다. langchain의 경우도 함수로 wrapping 되어 있지만, 그 속에는 OpenSearch API를 사용하고 있습니다.

OpenSearch를 구성하면, 웹으로 구현된 OpenSearch Dashboard에 접근할 수 있습니다. 이 곳에서 Dev tools를 사용하면 별도의 설정없이 바로 OpenSearch에 데이터를 넣고 검색해볼 수 있습니다.

## 3. 시작하기
간단한 것 부터 무작정 시작해 보겠습니다.

요청
```json
POST hello_opensearch/_doc
{
  "message": "hello world"
}
```

응답
```json
{
  "_index": "hello_opensearch", // 생성된 인덱스 명
  "_id": "r_b3tpUBvSYo7Ukfaqjm",  // 방금 등록된 문서 Id
  "_version": 1, // 문서 버전 번호
  "result": "created", //결과
  "_shards": {
    "total": 2, // 이 문서가 저장될 샤드 수 (Primary + Replica)
    "successful": 1, // 성공한 샤드 수
    "failed": 0
  },
  "_seq_no": 0, // 인덱스에 오퍼레이션이 있을 때마다 추가
  "_primary_term": 1 // failover과정에서 primary가 바뀔 때 증가하는 값
}
```

복제본 샤드에는 비동기로 저장됩니다.

#### 1. 인덱스 구성 확인

요청
```json
GET hello_opensearch
```

응답
```json
{
  "hello_opensearch": {
    "aliases": {},
    "mappings": { // 매핑정보
      "properties": {
        "message": { // 사용자가 지정한 필드 명
          "type": "text", // 필드 타입
          "fields": {
            "keyword": {
              "type": "keyword", // nested로 생성한 두번째 필드 타입
              "ignore_above": 256
            }
          }
        }
      }
    },
    "settings": {
      "index": {
        "replication": {
          "type": "DOCUMENT"
        },
        "number_of_shards": "5", // Primary 샤드 수
        "provided_name": "hello_opensearch",
        "creation_date": "1742531419189",
        "number_of_replicas": "1", // Replica 샤드 쌍
        "uuid": "Cz9SFU-oSc2K_s3kcWLbNg",
        "version": {
          "created": "136387827"
        }
      }
    }
  }
}
```

총 10개의 샤드를 구성하고, message라는 필드를 생성하면서 2가지의 타입으로 인덱싱하도록 구성했습니다.

#### 2. 문서번호를 지정해서 인덱싱

요청
```json
POST hello_opensearch/_doc/1
{
  "message": "first document"
}
```

> [!WARNING]
> 2025년 3월 21일 기준 OpenSearch Serverless의 Vector collection에서는 아직 문서 id를 지정해서 인덱싱하거나, 수정할 수 없습니다.

#### 3. 문서 번호로 검색
요청
```json
GET hello_opensearch/_doc/1
```

응답
```json
{
  "_index": "hello_opensearch",
  "_id": "1",
  "_version": 1,
  "_seq_no": 0,
  "_primary_term": 1,
  "found": true,
  "_source": {
    "message": "first document"
  }
}
```

자동으로 생성된 문서 id로 검색해 봅시다.

요청
```json
GET hello_opensearch/_doc/r_b3tpUBvSYo7Ukfaqjm
```

응답
```json
{
  "_index": "hello_opensearch",
  "_id": "r_b3tpUBvSYo7Ukfaqjm",
  "_version": 1,
  "_seq_no": 0,
  "_primary_term": 1,
  "found": true,
  "_source": {
    "message": "hello world"
  }
}
```

> [!WARNING]
> 문서번호를 균일하고(카디널리티가 높게) 중복없이 생성될 것이라고 확신이 없는 경우, 자동으로 생성되도록 문서 id를 직접 지정하지 마세요.


#### 4. 문서 수정
기존 문서id에 그대로 엎어쓰기

요청
```json
POST hello_opensearch/_doc/1
{
  "message": "first document_ver2"
}
```

응답
```json
{
  "_index": "hello_opensearch",
  "_id": "1",
  "_version": 2,
  "result": "updated",
  "_shards": {
    "total": 2,
    "successful": 1,
    "failed": 0
  },
  "_seq_no": 1,
  "_primary_term": 1
}
```

변경된 내용을 확인합니다.

요청
```json
GET hello_opensearch/_doc/1
```

응답
```json
{
  "_index": "hello_opensearch",
  "_id": "1",
  "_version": 2,
  "_seq_no": 1,
  "_primary_term": 1,
  "found": true,
  "_source": {
    "message": "first document_ver2"
  }
}
```

update API 사용하기 : https://opensearch.org/docs/latest/api-reference/document-apis/update-document/

요청
```json
POST hello_opensearch/_update/1
{
  "doc": {
    "message" : "first document_ver3"
  }
}
```

응답
```json
{
  "_index": "hello_opensearch",
  "_id": "1",
  "_version": 3,
  "result": "updated",
  "_shards": {
    "total": 2,
    "successful": 1,
    "failed": 0
  },
  "_seq_no": 2,
  "_primary_term": 1
}
```

_update를 사용하면 문서의 일부 필드만 업데이트를 할 수 있으며 스크립트를 활용하여 복잡한 업데이트 로직을 만들 수도 있습니다.

#### 5. 문서 검색

요청
```json
GET hello_opensearch/_search
```

응답
```json
{
  "took": 100,
  "timed_out": false,
  "_shards": {
    "total": 5,
    "successful": 5,
    "skipped": 0,
    "failed": 0
  },
  "hits": {
    "total": {
      "value": 2,
      "relation": "eq"
    },
    "max_score": 1,
    "hits": [
      {
        "_index": "hello_opensearch",
        "_id": "r_b3tpUBvSYo7Ukfaqjm",
        "_score": 1,
        "_source": {
          "message": "hello world"
        }
      },
      {
        "_index": "hello_opensearch",
        "_id": "1",
        "_score": 1,
        "_source": {
          "message": "first document_ver3"
        }
      }
    ]
  }
}
```

인덱스에 매핑 정보를 확인합니다.

요청
```json
GET hello_opensearch/_mapping
```

응답
```json
{
  "hello_opensearch": {
    "mappings": {
      "properties": {
        "message": {
          "type": "text",
          "fields": {
            "keyword": {
              "type": "keyword",
              "ignore_above": 256
            }
          }
        }
      }
    }
  }
}
```

match 쿼리를 이용해서 일부러 조금 다른 문장을 입력하고 검색합니다. match 쿼리는 검색어 텍스트에 분석기(Analyzer)를 적용합니다.

요청
```json
GET hello_opensearch/_search
{
  "query": {
    "match": {
      "message": "hello beautiful world"
    }
  }
}
```

응답
```json
{
  "took": 2,
  "timed_out": false,
  "_shards": {
    "total": 5,
    "successful": 5,
    "skipped": 0,
    "failed": 0
  },
  "hits": {
    "total": {
      "value": 1,
      "relation": "eq"
    },
    "max_score": 0.5753642,
    "hits": [
      {
        "_index": "hello_opensearch",
        "_id": "r_b3tpUBvSYo7Ukfaqjm",
        "_score": 0.5753642,
        "_source": {
          "message": "hello world"
        }
      }
    ]
  }
}
```

완전히 같은 문장이 아니어도 검색이 가능합니다. 그 이유는 조금 더 아래에서 살펴보겠습니다.

이번에는 term 쿼리를 이용해서 검색을 합니다. term 쿼리는 주어진 검색어를 그대로 사용하며, 분석기를 거치지 않습니다.

요청
```json
GET hello_opensearch/_search
{
  "query": {
    "term": {
      "message": "hello beautiful world"
    }
  }
}
```

응답
```json
{
  "took": 1,
  "timed_out": false,
  "_shards": {
    "total": 5,
    "successful": 5,
    "skipped": 0,
    "failed": 0
  },
  "hits": {
    "total": {
      "value": 0,
      "relation": "eq"
    },
    "max_score": null,
    "hits": []
  }
}
```

일반적으로 특정 단어나 카테고리를 필터링 하기 위해 keyword 필드를 사용합니다. 이 경우 인덱싱 할 때와, 검색할 때 모두 필터링이 타지 않도록 할 수 있습니다. 그래서 빠르게 필터링 하거나 true/false 정도의 수준의 검색이 필요할 때는 keyword로 매핑하고, term 쿼리로 검색합니다. 

#### 6. 인덱스 삭제

요청
```json
DELETE hello_opensearch
```

응답
```json
{
  "acknowledged": true
}
```

#### 7. 매핑 정보를 포함한 인덱스 생성

요청
```json
PUT hello_opensearch
{
  "mappings": {
    "properties": {
      "message": {
        "type": "text"
      },
      "from" : {
        "type": "keyword"
      }
    }
  }
}
```

응답
```json
{
  "acknowledged": true,
  "shards_acknowledged": true,
  "index": "hello_opensearch"
}
```

인덱스 정보 확인

요청
```json
GET hello_opensearch
```

응답
```json
{
  "hello_opensearch": {
    "aliases": {},
    "mappings": {
      "properties": {
        "from": {
          "type": "keyword"
        },
        "message": {
          "type": "text"
        }
      }
    },
    "settings": {
      "index": {
        "replication": {
          "type": "DOCUMENT"
        },
        "number_of_shards": "5",
        "provided_name": "hello_opensearch",
        "creation_date": "1742533287803",
        "number_of_replicas": "1",
        "uuid": "Hx4iU8UQQpa3mq9o-6N2mw",
        "version": {
          "created": "136387827"
        }
      }
    }
  }
}
```

매핑은 잘 되었지만, 샤드나 복제본의 수가 마음에 들지 않을 수 있습니다.

요청
```json
DELETE hello_opensearch
```

요청
```json
PUT hello_opensearch
{
  "settings": {
    "number_of_shards": "5",
    "number_of_replicas": "2"
  },
  "mappings": {
    "properties": {
      "message": {
        "type": "text"
      },
      "from" : {
        "type": "keyword"
      }
    }
  }
}
```
샤드의 수는 인덱스를 처음 생성할 때만 지정이 가능합니다. 복제본 수는 언제라도 수정할 수 있습니다.

자 이제, 가상의 상품 정보를 문서로 입력하고 검색하는 연습을 하겠습니다. 이 때 bulk 인덱스를 이용하여 효과적으로 데이터를 한번에 입력합니다.

요청
```json
PUT products
{
  "mappings" : {
    "properties": {
      "description": {
        "type": "text"
      },
      "category" : {
        "type": "keyword"
      }
    }
  }
}
```

응답
```json
{
  "acknowledged": true,
  "shards_acknowledged": true,
  "index": "products"
}
```

요청
```json
POST products/_bulk
{ "index" : { "_index" : "products" } }
{"description": "Experience crystal clear audio with our wireless earbuds that offer noise cancellation and long battery life.", "category": "Audio"}
{ "index" : { "_index" : "products" } }
{"description": "Capture stunning photos with our latest smartphone featuring a high-performance camera and all-day battery life.", "category": "Smartphone"}
{ "index" : { "_index" : "products" } }
{"description": "Stay productive on the go with our ultra-lightweight notebook that provides over 10 hours of battery life.", "category": "Computer"}
{ "index" : { "_index" : "products" } }
{"description": "Immerse yourself in vibrant colors and sharp details with our 55-inch 4K OLED TV.", "category": "TV"}
{ "index" : { "_index" : "products" } }
{"description": "Control your entire smart home ecosystem with voice commands using our intuitive Smart Home Hub.", "category": "Smart Home"}
{ "index" : { "_index" : "products" } }
{"description": "Keep your cards and cash safe with our premium leather wallet featuring RFID blocking technology.", "category": "Accessories"}
{ "index" : { "_index" : "products" } }
{"description": "Let our multifunctional robot vacuum handle both mopping and dust suction for a spotless home.", "category": "Home Appliance"}
{ "index" : { "_index" : "products" } }
{"description": "Enjoy barista-quality coffee at home with our premium coffee machine that makes everything from espresso to latte.", "category": "Kitchen Appliance"}
{ "index" : { "_index" : "products" } }
{"description": "Capture breathtaking aerial footage with our high-performance camera drone featuring extended flight time.", "category": "Electronics"}
{ "index" : { "_index" : "products" } }
{"description": "Game in comfort for hours with our ergonomically designed gaming chair, perfect for extended gaming sessions.", "category": "Furniture"}
```

응답
```json
{
  "took": 260,
  "errors": false,
  "items": [
    {
      "index": {
        "_index": "products",
        "_id": "1%3A0%3ADfn7s5UBA7vOKC2CDSnR",
        "_version": 1,
        "result": "created",
        "_shards": {
          "total": 0,
          "successful": 0,
          "failed": 0
        },
        "_seq_no": 0,
        "_primary_term": 0,
        "status": 201
      }
    },
    {
      "index": {
        "_index": "products",
        "_id": "1%3A0%3ADvn7s5UBA7vOKC2CDSnR",
        "_version": 1,
        "result": "created",
        "_shards": {
          "total": 0,
          "successful": 0,
          "failed": 0
        },
        "_seq_no": 0,
        "_primary_term": 0,
        "status": 201
      }
    },
    {
      "index": {
        "_index": "products",
        "_id": "1%3A0%3AD_n7s5UBA7vOKC2CDSnR",
        "_version": 1,
        "result": "created",
        "_shards": {
          "total": 0,
          "successful": 0,
          "failed": 0
        },
        "_seq_no": 0,
        "_primary_term": 0,
        "status": 201
      }
    },
    {
      "index": {
        "_index": "products",
        "_id": "1%3A0%3AEPn7s5UBA7vOKC2CDSnR",
        "_version": 1,
        "result": "created",
        "_shards": {
          "total": 0,
          "successful": 0,
          "failed": 0
        },
        "_seq_no": 0,
        "_primary_term": 0,
        "status": 201
      }
    },
    {
      "index": {
        "_index": "products",
        "_id": "1%3A0%3AEfn7s5UBA7vOKC2CDSnR",
        "_version": 1,
        "result": "created",
        "_shards": {
          "total": 0,
          "successful": 0,
          "failed": 0
        },
        "_seq_no": 0,
        "_primary_term": 0,
        "status": 201
      }
    },
    {
      "index": {
        "_index": "products",
        "_id": "1%3A0%3AEvn7s5UBA7vOKC2CDSnR",
        "_version": 1,
        "result": "created",
        "_shards": {
          "total": 0,
          "successful": 0,
          "failed": 0
        },
        "_seq_no": 0,
        "_primary_term": 0,
        "status": 201
      }
    },
    {
      "index": {
        "_index": "products",
        "_id": "1%3A0%3AE_n7s5UBA7vOKC2CDSnR",
        "_version": 1,
        "result": "created",
        "_shards": {
          "total": 0,
          "successful": 0,
          "failed": 0
        },
        "_seq_no": 0,
        "_primary_term": 0,
        "status": 201
      }
    },
    {
      "index": {
        "_index": "products",
        "_id": "1%3A0%3AFPn7s5UBA7vOKC2CDSnS",
        "_version": 1,
        "result": "created",
        "_shards": {
          "total": 0,
          "successful": 0,
          "failed": 0
        },
        "_seq_no": 0,
        "_primary_term": 0,
        "status": 201
      }
    },
    {
      "index": {
        "_index": "products",
        "_id": "1%3A0%3AFfn7s5UBA7vOKC2CDSnS",
        "_version": 1,
        "result": "created",
        "_shards": {
          "total": 0,
          "successful": 0,
          "failed": 0
        },
        "_seq_no": 0,
        "_primary_term": 0,
        "status": 201
      }
    },
    {
      "index": {
        "_index": "products",
        "_id": "1%3A0%3AFvn7s5UBA7vOKC2CDSnS",
        "_version": 1,
        "result": "created",
        "_shards": {
          "total": 0,
          "successful": 0,
          "failed": 0
        },
        "_seq_no": 0,
        "_primary_term": 0,
        "status": 201
      }
    }
  ]
}
```

아래의 문장으로 검색해보겠습니다. 이 문장과 완전히 동일한 설명은 없는 상황입니다.

요청
```json
POST /products/_search
{
  "query": {
    "match": {
      "description": "long battery life products"
    }
  }
}
```

응답
```json
{
  "took": 19,
  "timed_out": false,
  "_shards": {
    "total": 0,
    "successful": 0,
    "skipped": 0,
    "failed": 0
  },
  "hits": {
    "total": {
      "value": 3,
      "relation": "eq"
    },
    "max_score": 2.9267392,
    "hits": [
      {
        "_index": "products",
        "_id": "1%3A0%3A4NgAtJUB5fVSzNcdd8LE",
        "_score": 2.9267392,
        "_source": {
          "description": "Experience crystal clear audio with our wireless earbuds that offer noise cancellation and long battery life.",
          "category": "Audio"
        }
      },
      {
        "_index": "products",
        "_id": "1%3A0%3A4dgAtJUB5fVSzNcdd8LG",
        "_score": 1.351733,
        "_source": {
          "description": "Capture stunning photos with our latest smartphone featuring a high-performance camera and all-day battery life.",
          "category": "Smartphone"
        }
      },
      {
        "_index": "products",
        "_id": "1%3A0%3A4tgAtJUB5fVSzNcdd8LG",
        "_score": 1.318853,
        "_source": {
          "description": "Stay productive on the go with our ultra-lightweight notebook that provides over 10 hours of battery life.",
          "category": "Computer"
        }
      }
    ]
  }
}
```

그럼에도, term을 기준으로 BM25 알고리즘에 의해 _score가 생성되고 결과가 나타납니다.

아래는 문장을 검색하면서 스마트폰인 경우만 필터링하고 싶은 경우 입니다. 이때 category가 keyword 타입인 것을 확인하세요.

요청
```json
POST /products/_search
{
  "query": {
    "bool": {
      "must": {
        "match": {
          "description": "long battery life products"
        }
      },
      "filter": {
        "term": {
          "category": "Smartphone"
        }
      }
    }
  }
}
```

응답
```json
{
  "took": 20,
  "timed_out": false,
  "_shards": {
    "total": 0,
    "successful": 0,
    "skipped": 0,
    "failed": 0
  },
  "hits": {
    "total": {
      "value": 1,
      "relation": "eq"
    },
    "max_score": 1.351733,
    "hits": [
      {
        "_index": "products",
        "_id": "1%3A0%3A4dgAtJUB5fVSzNcdd8LG",
        "_score": 1.351733,
        "_source": {
          "description": "Capture stunning photos with our latest smartphone featuring a high-performance camera and all-day battery life.",
          "category": "Smartphone"
        }
      }
    ]
  }
}
```

## 분석기 확인
https://opensearch.org/docs/latest/analyzers/supported-analyzers/index/

_analyze API를 이용해서, 내가 입력한 문장이 어떻게 분석되어 term으로 분리되는지 알 수 있습니다.

요청
```json
GET _analyze
{
  "analyzer": "standard",
  "text": "long battery life products"
}
```

응답
```json
{
  "tokens": [
    {
      "token": "long",
      "start_offset": 0,
      "end_offset": 4,
      "type": "<ALPHANUM>",
      "position": 0
    },
    {
      "token": "battery",
      "start_offset": 5,
      "end_offset": 12,
      "type": "<ALPHANUM>",
      "position": 1
    },
    {
      "token": "life",
      "start_offset": 13,
      "end_offset": 17,
      "type": "<ALPHANUM>",
      "position": 2
    },
    {
      "token": "products",
      "start_offset": 18,
      "end_offset": 26,
      "type": "<ALPHANUM>",
      "position": 3
    }
  ]
}
```

기본 값으로 standard 분석기가 사용되지만, 다양한 분석기를 통해 검색을 고도화할 수 있습니다.

## 한글 연습
한글의 경우 Nori 분석기를 사용하면 더 검색 품질이 높아집니다. 아래 예제를 참고하세요.

- 참고: https://aws.amazon.com/ko/blogs/tech/amazon-opensearch-service-korean-nori-plugin-for-analysis/


## 벡터 검색
요청
```json
PUT my-index
{
  "settings": {
    "index.knn": true
  },
  "mappings": {
    "properties": {
      "my_vector1": {
        "type": "knn_vector",
        "dimension": 2
      },
      "my_vector2": {
        "type": "knn_vector",
        "dimension": 4
      }
    }
  }
}
```

요청
```json
POST _bulk
{ "index": { "_index": "my-index" } }
{ "my_vector1": [1.5, 2.5], "price": 12.2 }
{ "index": { "_index": "my-index" } }
{ "my_vector1": [2.5, 3.5], "price": 7.1 }
{ "index": { "_index": "my-index" } }
{ "my_vector1": [3.5, 4.5], "price": 12.9 }
{ "index": { "_index": "my-index" } }
{ "my_vector1": [5.5, 6.5], "price": 1.2 }
{ "index": { "_index": "my-index" } }
{ "my_vector1": [4.5, 5.5], "price": 3.7 }
{ "index": { "_index": "my-index" } }
{ "my_vector2": [1.5, 5.5, 4.5, 6.4], "price": 10.3 }
{ "index": { "_index": "my-index" } }
{ "my_vector2": [2.5, 3.5, 5.6, 6.7], "price": 5.5 }
{ "index": { "_index": "my-index" } }
{ "my_vector2": [4.5, 5.5, 6.7, 3.7], "price": 4.4 }
{ "index": { "_index": "my-index" } }
{ "my_vector2": [1.5, 5.5, 4.5, 6.4], "price": 8.9 }
```

아래와 같이 벡터 겁색을 수행할 수 있습니다.

요청
```json
GET my-index/_search
{
  "size": 2,
  "query": {
    "knn": {
      "my_vector2": {
        "vector": [2, 3, 5, 6],
        "k": 2
      }
    }
  }
}
```

특히 가장 가까운 k=2를 검색하면서, 클러스터 레벨에서 최종 결과를 2개만 반환하도록 합니다.

응답
```json
{
  "took": 14,
  "timed_out": false,
  "_shards": {
    "total": 0,
    "successful": 0,
    "skipped": 0,
    "failed": 0
  },
  "hits": {
    "total": {
      "value": 2,
      "relation": "eq"
    },
    "max_score": 0.12642226,
    "hits": [
      {
        "_index": "my-index",
        "_id": "1%3A0%3AKvk8tJUBA7vOKC2C_Skc",
        "_score": 0.12642226,
        "_source": {
          "my_vector2": [
            1.5,
            5.5,
            4.5,
            6.4
          ],
          "price": 8.9
        }
      },
      {
        "_index": "my-index",
        "_id": "1%3A0%3AKfk8tJUBA7vOKC2C_Skc",
        "_score": 0.04612546,
        "_source": {
          "my_vector2": [
            4.5,
            5.5,
            6.7,
            3.7
          ],
          "price": 4.4
        }
      }
    ]
  }
}
```

필터 기능을 이용해서 벡터 값 외의 필드를 기준으로 필터링할 수도 있습니다. 이 예제에서는 상품의 의미가 my_vector2에 저장되어 있고, 가격이 price 필드에 저장되어 있습니다. 그래서 의미론 적으로 유사한 2개를 찾고, 결과에서는 가격이 6이상 10이하인 것만 받도록 규정합니다.

요청
```json
GET my-index/_search
{
  "size": 2,
  "query": {
    "knn": {
      "my_vector2": {
        "vector": [2, 3, 5, 6],
        "k": 2
      }
    }
  },
  "post_filter": {
    "range": {
      "price": {
        "gte": 6,
        "lte": 10
      }
    }
  }
}
```

응답
```json
{
  "took": 14,
  "timed_out": false,
  "_shards": {
    "total": 0,
    "successful": 0,
    "skipped": 0,
    "failed": 0
  },
  "hits": {
    "total": {
      "value": 1,
      "relation": "eq"
    },
    "max_score": 0.12642226,
    "hits": [
      {
        "_index": "my-index",
        "_id": "1%3A0%3AKvk8tJUBA7vOKC2C_Skc",
        "_score": 0.12642226,
        "_source": {
          "my_vector2": [
            1.5,
            5.5,
            4.5,
            6.4
          ],
          "price": 8.9
        }
      }
    ]
  }
}
```

자 여기까지 기본 OpenSearch의 사용방법을 알아봤습니다. 벡터 검색에 대한 더 깊은 내용은 세션자료와 이후 Lab에서 심화학습 합니다.

## 기타 주제
### 1. ML 플러그인 활용

ML 플러그인을 활용하면, 외부에서 임베딩 파이프라인을 구현하지 않아도 OpenSearch에 데이터를 저장하는 시점에 한번에 임베딩 데이터를 만들고 인덱싱할 수 있습니다.

- Amazon OpenSearch Service의 AI/ML 커넥터로 Neural 검색 강화
  - https://aws.amazon.com/ko/blogs/tech/power-neural-search-with-ai-ml-connectors-in-amazon-opensearch-service/


### 2. ANN 알고리즘의 파라메터와 라이브러리

- Methods and engines
  - https://opensearch.org/docs/latest/field-types/supported-field-types/knn-methods-engines/
- 파라미터의 수치별 성능, 메모리 사용량, recall의 상관관계에 대한 테스트는 아래의 블로그를 참조하세요
  - https://aws.amazon.com/ko/blogs/tech/choose-the-k-nn-algorithm-for-your-billion-scale-use-case-with-opensearch/

### 3. 벡터 검색시 메모리 사용량

- ANN으로 벡터를 생성 후 warm up API를 이용해서 오프힙에 벡터와 관련된 데이터를 모두 올려놓는 것을 추천드립니다.
  - https://opensearch.org/docs/latest/vector-search/api/#warmup-operation
- 메모리 사용량의 계산은 데이터 노드 인스턴스 메모리의 * 0.5 * 0.5가 가용 가능한 최대 메모리 입니다.
  - https://docs.aws.amazon.com/ko_kr/opensearch-service/latest/developerguide/knn.html#knn-settings
- Stat API를 이용해서, 현재 오프힙 메모리에 벡터가 얼마나 올라와 있는지, 디스크 I/O 발생량은 얼마인지, 메모리 증설이 필요한지 예측할 수 있습니다.
  - https://opensearch.org/docs/latest/vector-search/api/#stats
- 수억건 이상의 데이터에 벡터검색이 필요한 경우 비용 절감을 위해 Disk-based search도 고려할 수 있습니다.
  - https://opensearch.org/docs/latest/vector-search/optimizing-storage/disk-based-vector-search/

