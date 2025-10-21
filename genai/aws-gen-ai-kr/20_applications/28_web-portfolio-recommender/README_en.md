# Bedrock Web Portfolio Analyzer

Web portfolio image analysis and user preference matching tool using Amazon Bedrock

## Overview

This project is a tool that comprehensively analyzes web portfolio images using Amazon Bedrock's multimodal AI model (Amazon Nova Pro) and quantitatively evaluates compatibility with user-defined preference keywords.

## Key Features

### 1. Portfolio Image Analysis
- Website design, layout, and color scheme analysis
- User experience (UX) and accessibility evaluation
- Technical characteristics and industry suitability analysis
- Detailed analysis report generation including strengths and improvement points

### 2. User Preference Matching
- Evaluate compatibility between user-defined keywords and portfolio with 0-100% scoring
- Keyword-specific relevance analysis and comprehensive recommendation provision
- Detailed explanation of compatibility rationale and recommendation reasons

### 3. Automatic Image Compression
- Automatically compress images to meet Bedrock API limits (5MB)
- Automatic optimization of balance between quality and file size
- Provide original image size and compression result information

## Model Used
- **Amazon Nova Pro** (`us.amazon.nova-pro-v1:0`): Multimodal image analysis and text generation

## Project Structure

```
web-portfolio-recommender/
├── portfolio/
│   ├── skincare.png           # Skincare website portfolio sample
│   └── tech.png              # Tech company website portfolio sample
├── analysis-and-recommendation.ipynb  # Main analysis notebook
└── README.md
```

## Analysis Categories

### 1. Design Analysis
- **Layout Structure**: Arrangement and composition of header, navigation, main content, sidebar, footer
- **Color Scheme**: Primary colors, secondary colors, color combination characteristics and brand consistency
- **Typography**: Font types, sizes, hierarchy, comprehensive readability analysis
- **Visual Elements**: Style and utilization of icons, images, graphic elements
- **White Space**: Use of margins and overall balance

### 2. Function and Structure Analysis
- **Navigation**: Menu structure, user paths, accessibility evaluation
- **Content Organization**: Information structuring, content prioritization, page flow
- **Interactive Elements**: Design and placement of buttons, links, form elements
- **Responsive Design**: Adaptability inference for various screen sizes

### 3. User Experience (UX) Analysis
- **Readability**: Ease of reading text and content
- **Intuitiveness**: Interface composition that users can easily understand
- **Accessibility**: Design elements considering accessibility for people with disabilities
- **User Goals**: Main purpose of website and effectiveness of user behavior induction

### 4. Technical Feature Inference
- **Development Technology**: Estimated frontend technologies or frameworks used
- **Performance Optimization**: Image optimization, loading speed considerations
- **Browser Compatibility**: Cross-browsing considerations

### 5. Industry and Purpose Analysis
- **Target Industry**: Industry or field targeted by the portfolio
- **Business Goals**: Analysis of main business purposes of the website
- **Target Users**: Analysis of main user groups and their needs

## Usage

### 1. Environment Setup
```python
import boto3
region = "us-west-2"
nova_pro_model_id = "us.amazon.nova-pro-v1:0"

client = boto3.client(service_name="bedrock-runtime", region_name=region)
```

### 2. Portfolio Image Preparation
- Place image files to analyze in `portfolio/` folder (Sample images used are from imweb's expert finding service portfolio)
- Supported formats: PNG, JPEG, PDF
- Image size: Automatically compressed to under 5MB

### 3. Basic Analysis Execution
```python
# Set target image filename
target = "skincare.png"  # or "tech.png"

# Execute basic analysis
main()
```

### 4. User Preference-based Compatibility Analysis
```python
# Define user preference keywords
user_keywords = ["beauty", "MZ generation", "skincare cosmetics", "makeup"]
# or
user_keywords = ["AI", "IT", "professionalism"]

# Execute enhanced analysis (basic analysis + compatibility analysis)
enhanced_main()
```

## Analysis Result Examples

### Skincare Portfolio Analysis
**User Keywords**: ["beauty", "MZ generation", "skincare cosmetics", "makeup"]
- **Compatibility Score**: 88-95%
- **Key Features**: 
  - Pink/beige color scheme creating feminine brand image
  - Product-centered visual layout
  - Modern and clean design targeting MZ generation
- **Recommendation**: Strongly recommended

### Tech Portfolio Analysis
**User Keywords**: ["AI", "IT", "professionalism"]
- **Compatibility Score**: 75-90%
- **Key Features**:
  - Professional and trustworthy colors based on blue/black
  - Structural layout emphasizing services and technical capabilities
  - Professionalism-centered design targeting B2B customers
- **Recommendation**: Recommended

## Detailed Key Features

### Automatic Image Compression System
- **Compression Algorithm**: Quality-step compression → Size adjustment → Final optimization
- **Compression Process**: 
  1. Step-by-step compression from 95% to 10% quality
  2. Size adjustment (80% → 60% → 40% → 20%)
  3. Final compression to 800x600 resolution
- **Result Information**: Pre/post compression size, final quality, resolution information

### Token Usage Monitoring
- **Cache System**: Token usage optimization through system prompt caching
- **Usage Tracking**: Detailed information on input/output tokens, cache creation/read tokens
- **Cost Management**: Step-by-step token usage and total usage display

## Requirements

### System Requirements
- **Python**: 3.12+
- **Required Libraries**:
  - `boto3`: AWS SDK
  - `PIL (Pillow)`: Image processing
  - `io`, `os`: File and memory management

### AWS Configuration
- **Amazon Bedrock access permissions** required
- **Supported Region**: us-west-2 (default setting)
- **Required Model Access**:
  - Amazon Nova Pro (`us.amazon.nova-pro-v1:0`)

## Precautions and Limitations
- **Image Size**: Automatically compressed to under 5MB (Bedrock API limitation)
- **Analysis Accuracy**: Based on AI model inference, use for reference purposes
- **Cost Management**: Cost optimization through token usage monitoring required
- **Supported Formats**: PNG, JPEG, PDF (PDF requires conversion to image)
- **Language**: Korean-based analysis and results provided