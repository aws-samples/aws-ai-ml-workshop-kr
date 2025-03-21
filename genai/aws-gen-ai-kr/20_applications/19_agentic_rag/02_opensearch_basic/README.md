# OpenSearch 기본 연습

## 1. 설치

## 2. Dev tools
이 워크샵은 주로 파이썬 코드와 langchain을 이용하여 OpenSearch와 통신하게 됩니다. 하지만 그 전에 Lab2에서는 순수한 OpenSearch API만을 사용하여 기본적인 이해를 돕습니다. langchain의 경우도 함수로 wrapping 되어 있지만, 그 속에는 OpenSearch API를 사용하고 있습니다.

OpenSearch를 구성하면, 웹으로 구현된 OpenSearch Dashboard에 접근할 수 있습니다. 이 곳에서 Dev tools를 사용하면 별도의 설정없이 바로 OpenSearch에 데이터를 넣고 검색해볼 수 있습니다.

## 3. 시작하기

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
  "_index": "hello_opensearch",
  "_id": "1%3A0%3AF_kKtJUBA7vOKC2C1imt",
  "_version": 1,
  "result": "created",
  "_shards": {
    "total": 0,
    "successful": 0,
    "failed": 0
  },
  "_seq_no": 0,
  "_primary_term": 0
}
```

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
    },
    "settings": {
      "index": {
        "creation_date": "1742479529101",
        "number_of_shards": "2",
        "number_of_replicas": "0",
        "uuid": "utnfs5UBvhSoiboKlGg_",
        "version": {
          "created": "136327827"
        },
        "provided_name": "hello_opensearch"
      }
    }
  }
}
```

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

업데이트, 딜리트, 특정 아이디 값 가져오기 등

가상의 상품 정보를 문서로 입력하고, 검색하는 연습을 하겠습니다. 이 때 bulk 인덱스를 이용하여 효과적으로 데이터를 한번에 입력합니다.

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

## 한글 연습
참고: https://aws.amazon.com/ko/blogs/tech/amazon-opensearch-service-korean-nori-plugin-for-analysis/


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

답이 바로 안나오는 경우

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