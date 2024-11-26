# AICC - 자동차 보험상담 워크샵

이 워크샵에서는 인공지능을 활용한 고객 상담 자동화 시스템인 AICC(Artificial Intelligence Call Center)를 구현하는 방법을 학습합니다. 참가자들은 프롬프트 엔지니어링을 통해 자동차 보험 상담 시나리오를 해결하게 됩니다.

> **중요**: 이 저장소는 실습 코드만을 포함하고 있습니다. 실제 워크샵 수행과 상세한 가이드를 위해서는 아래 링크의 전체 워크샵 페이지를 참조해 주세요.
> 
> [AICC - 자동차 보험상담 워크샵](https://catalog.us-east-1.prod.workshops.aws/workshops/86f59566-0ae7-44be-80ab-9044b83c88f2/ko-KR/prompting)

## 개요

AICC는 인공지능을 활용하여 고객 상담을 자동화하고, 보다 효율적이고 정확한 상담 서비스를 제공하는 시스템입니다. 이 워크샵에서는 Claude 모델을 사용하여 AICC 애플리케이션을 구현하고, 프롬프트 엔지니어링을 통해 다양한 상담 시나리오를 해결합니다.

## 폴더 구조

- `practice`: 실습을 직접 수행할 수 있는 코드가 포함된 폴더입니다. 참가자들은 이 폴더의 파일들을 수정하며 실습을 진행합니다.
- `completed`: 각 실습의 완성된 코드가 포함된 폴더입니다. 참고용으로 사용할 수 있습니다.

## AICC 시나리오

1. **상담 요약**: 녹취된 상담의 상세한 요약 생성
2. **상담 노트**: 상담 기록에서 중요 정보 추출 및 노트 작성
3. **메일 회신**: 고객 문의에 대한 자동 답변 생성
4. **상담 품질**: 녹취록 기반 상담 품질 자동 평가

## 프롬프트 작성 가이드

프롬프트는 `prompting/practice/` 디렉토리 내의 각 시나리오별 텍스트 파일에서 작성합니다:

- 상담 요약: `summary.txt`
- 상담 노트: `note.txt`
- 메일 회신: `reply.txt`
- 상담 품질: `quality.txt`

이 워크샵을 통해 참가자들은 프롬프트 엔지니어링 기술을 실제 비즈니스 시나리오에 적용하는 방법을 학습하고, AI 기반 고객 상담 시스템의 구현 및 최적화 능력을 향상시킬 수 있습니다.

---

# AICC - Automotive Insurance Consultation Workshop

This workshop teaches how to implement an AICC (Artificial Intelligence Call Center), an automated customer consultation system using artificial intelligence. Participants will solve automotive insurance consultation scenarios through prompt engineering.

> **Important**: This repository contains only the practice code. For the actual workshop execution and detailed guide, please refer to the full workshop page at the link below.
> 
> [AICC - Automotive Insurance Consultation Workshop](https://catalog.us-east-1.prod.workshops.aws/workshops/86f59566-0ae7-44be-80ab-9044b83c88f2/en-US/prompting)

## Overview

AICC is a system that automates customer consultations using artificial intelligence, providing more efficient and accurate consultation services. In this workshop, we will implement an AICC application using the Claude model and solve various consultation scenarios through prompt engineering.

## Folder Structure

- `practice`: This folder contains code for hands-on exercises. Participants will modify files in this folder during the workshop.
- `completed`: This folder contains the completed code for each exercise. It can be used for reference.

## AICC Scenarios

1. **Consultation Summary**: Generate a detailed summary of recorded consultations
2. **Consultation Notes**: Extract important information from consultation records and create notes
3. **Email Reply**: Automatically generate responses to customer inquiries
4. **Consultation Quality**: Automatically evaluate consultation quality based on transcripts

## Prompt Writing Guide

Prompts are written in scenario-specific text files within the `prompting/practice/` directory:

- Consultation Summary: `summary.txt`
- Consultation Notes: `note.txt`
- Email Reply: `reply.txt`
- Consultation Quality: `quality.txt`

Through this workshop, participants will learn how to apply prompt engineering techniques to real business scenarios and improve their ability to implement and optimize AI-based customer consultation systems.
