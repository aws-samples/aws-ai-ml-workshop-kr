import streamlit as st
import boto3
import json
import base64
from PIL import Image, ImageDraw, ImageFont
import io
import numpy as np
from streamlit_drawable_canvas import st_canvas
import math
from datetime import datetime

# AWS 클라이언트 초기화
@st.cache_resource
def init_aws_clients():
    try:
        session = boto3.Session()
        bedrock = session.client('bedrock-runtime', region_name='us-east-1')
        return bedrock
    except Exception as e:
        st.error(f"AWS 클라이언트 초기화 실패: {str(e)}")
        return None

def get_korean_font(size=20):
    """한글을 지원하는 폰트를 찾아서 반환"""
    korean_fonts = [
        "/System/Library/Fonts/AppleSDGothicNeo.ttc",
        "/System/Library/Fonts/Supplemental/AppleGothic.ttf",
        "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
    ]
    
    for font_path in korean_fonts:
        try:
            return ImageFont.truetype(font_path, size)
        except:
            continue
    
    # 모든 폰트가 실패하면 기본 폰트 사용
    try:
        return ImageFont.load_default()
    except:
        return None

def translate_to_english(korean_text, bedrock_client):
    """Nova Pro를 사용하여 한국어를 영어로 번역"""
    if bedrock_client is None:
        return korean_text
        
    try:
        prompt = f"Translate the following Korean text to English. Only provide the translation without any additional text:\n\n{korean_text}"
        
        body = json.dumps({
            "messages": [
                {
                    "role": "user",
                    "content": [{"text": prompt}]
                }
            ],
            "inferenceConfig": {
                "maxTokens": 1000,
                "temperature": 0.1
            }
        })
        
        response = bedrock_client.invoke_model(
            body=body,
            modelId="amazon.nova-pro-v1:0",
            accept="application/json",
            contentType="application/json"
        )
        
        response_body = json.loads(response.get('body').read())
        return response_body['output']['message']['content'][0]['text'].strip()
    except Exception as e:
        st.error(f"번역 중 오류가 발생했습니다: {str(e)}")
        return korean_text

def generate_menuboard_image(prompt, bedrock_client):
    """Nova Canvas를 사용하여 메뉴보드 이미지 생성"""
    if bedrock_client is None:
        return None
        
    try:
        body = json.dumps({
            "taskType": "TEXT_IMAGE",
            "textToImageParams": {
                "text": prompt,
                "negativeText": "blurry, low quality, distorted, watermark"
            },
            "imageGenerationConfig": {
                "numberOfImages": 1,
                "height": 768,
                "width": 1024,
                "cfgScale": 8.0,
                "seed": 42
            }
        })
        
        response = bedrock_client.invoke_model(
            body=body,
            modelId="amazon.nova-canvas-v1:0",
            accept="application/json",
            contentType="application/json"
        )
        
        response_body = json.loads(response.get('body').read())
        
        if 'images' in response_body and len(response_body['images']) > 0:
            image_data = base64.b64decode(response_body['images'][0])
            return Image.open(io.BytesIO(image_data))
        else:
            st.error("이미지 생성에 실패했습니다.")
            return None
            
    except Exception as e:
        st.error(f"이미지 생성 중 오류가 발생했습니다: {str(e)}")
        return None

def calculate_menu_positions(text_box, menus, layout_type="grid"):
    """텍스트 박스 내에서 메뉴들의 위치를 자동 계산"""
    x, y, width, height = text_box
    menu_count = len(menus)
    
    if menu_count == 0:
        return []
    
    positions = []
    padding = 10
    
    if layout_type == "grid":
        # 격자 형태로 배치
        cols = math.ceil(math.sqrt(menu_count))
        rows = math.ceil(menu_count / cols)
        
        item_width = max(50, (width - padding * (cols + 1)) // cols)
        item_height = max(30, (height - padding * (rows + 1)) // rows)
        
        for i, menu in enumerate(menus):
            row = i // cols
            col = i % cols
            
            item_x = x + padding + col * (item_width + padding)
            item_y = y + padding + row * (item_height + padding)
            
            position = {
                'menu_name': menu['name'],
                'price': menu['price'],
                'bbox': (item_x, item_y, item_width, item_height),
                'alignment': menu.get('alignment', 'center'),
                'use_auto_font': menu.get('use_auto_font', True)
            }
            
            if not menu.get('use_auto_font', True) and 'font_size' in menu:
                position['font_size'] = menu['font_size']
            
            positions.append(position)
    
    elif layout_type == "vertical":
        # 세로로 나열
        item_height = max(30, (height - padding * (menu_count + 1)) // menu_count)
        item_width = width - padding * 2
        
        for i, menu in enumerate(menus):
            item_x = x + padding
            item_y = y + padding + i * (item_height + padding)
            
            position = {
                'menu_name': menu['name'],
                'price': menu['price'],
                'bbox': (item_x, item_y, item_width, item_height),
                'alignment': menu.get('alignment', 'left'),
                'use_auto_font': menu.get('use_auto_font', True)
            }
            
            if not menu.get('use_auto_font', True) and 'font_size' in menu:
                position['font_size'] = menu['font_size']
            
            positions.append(position)
    
    elif layout_type == "horizontal":
        # 가로로 나열
        item_width = max(100, (width - padding * (menu_count + 1)) // menu_count)
        item_height = height - padding * 2
        
        for i, menu in enumerate(menus):
            item_x = x + padding + i * (item_width + padding)
            item_y = y + padding
            
            position = {
                'menu_name': menu['name'],
                'price': menu['price'],
                'bbox': (item_x, item_y, item_width, item_height),
                'alignment': menu.get('alignment', 'center'),
                'use_auto_font': menu.get('use_auto_font', True)
            }
            
            if not menu.get('use_auto_font', True) and 'font_size' in menu:
                position['font_size'] = menu['font_size']
            
            positions.append(position)
    
    return positions

def get_optimal_font_size(draw, text, max_width, max_height, max_font_size=50, min_font_size=8):
    """텍스트가 지정된 영역에 맞는 최적의 폰트 크기를 찾음"""
    for font_size in range(max_font_size, min_font_size - 1, -1):
        font = get_korean_font(font_size)
        if font is None:
            continue
            
        try:
            bbox = draw.textbbox((0, 0), text, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            
            if text_width <= max_width and text_height <= max_height:
                return font, font_size
        except:
            continue
    
    # 최소 폰트 크기로도 안 맞으면 최소 크기 반환
    return get_korean_font(min_font_size), min_font_size

def wrap_text(text, font, max_width, draw):
    """텍스트를 지정된 너비에 맞게 줄바꿈"""
    words = text.split()
    lines = []
    current_line = ""
    
    for word in words:
        test_line = current_line + (" " if current_line else "") + word
        try:
            bbox = draw.textbbox((0, 0), test_line, font=font)
            line_width = bbox[2] - bbox[0]
        except:
            line_width = len(test_line) * 10
        
        if line_width <= max_width:
            current_line = test_line
        else:
            if current_line:
                lines.append(current_line)
                current_line = word
            else:
                lines.append(word)
    
    if current_line:
        lines.append(current_line)
    
    return lines

def add_text_overlay(image, menu_positions, font_size_override=None):
    """이미지에 텍스트 오버레이 추가 (개선된 버전)"""
    try:
        img_copy = image.copy()
        
        if get_korean_font(20) is None:
            st.warning("폰트를 로드할 수 없습니다.")
            return img_copy
        
        # 각 메뉴 아이템 그리기
        for position in menu_positions:
            try:
                x, y, width, height = position['bbox']
                menu_name = str(position.get('menu_name', ''))
                price = str(position.get('price', ''))
                alignment = position.get('alignment', 'center')
                use_auto_font = position.get('use_auto_font', True)
                custom_font_size = position.get('font_size', font_size_override)
                
                # 개별 메뉴 배경
                overlay = Image.new('RGBA', img_copy.size, (255, 255, 255, 0))
                overlay_draw = ImageDraw.Draw(overlay)
                overlay_draw.rectangle([x, y, x + width, y + height], 
                                     fill=(255, 255, 255, 200), 
                                     outline=(200, 200, 200, 255), width=1)
                img_copy = Image.alpha_composite(img_copy.convert('RGBA'), overlay).convert('RGB')
                
                # 텍스트 그리기
                draw = ImageDraw.Draw(img_copy)
                
                # 여백 설정
                padding = 5
                text_area_width = width - (padding * 2)
                text_area_height = height - (padding * 2)
                
                # 폰트 크기 결정
                if use_auto_font:
                    # 자동 폰트 크기 조절
                    combined_text = f"{menu_name} {price}"
                    menu_font, font_size = get_optimal_font_size(
                        draw, combined_text, text_area_width, text_area_height // 2
                    )
                    price_font = get_korean_font(max(8, font_size - 4))
                elif custom_font_size:
                    # 사용자 지정 폰트 크기
                    menu_font = get_korean_font(custom_font_size)
                    price_font = get_korean_font(max(8, custom_font_size - 4))
                else:
                    # 기본 폰트 크기
                    menu_font = get_korean_font(20)
                    price_font = get_korean_font(16)
                
                if menu_font is None or price_font is None:
                    continue
                
                # 메뉴명 줄바꿈 처리
                menu_lines = wrap_text(menu_name, menu_font, text_area_width, draw)
                price_lines = wrap_text(price, price_font, text_area_width, draw)
                
                # 텍스트 높이 계산
                try:
                    menu_line_height = draw.textbbox((0, 0), "A", font=menu_font)[3] - draw.textbbox((0, 0), "A", font=menu_font)[1]
                    price_line_height = draw.textbbox((0, 0), "A", font=price_font)[3] - draw.textbbox((0, 0), "A", font=price_font)[1]
                except:
                    menu_line_height = custom_font_size if custom_font_size else 20
                    price_line_height = (custom_font_size - 4) if custom_font_size else 16
                
                total_menu_height = len(menu_lines) * menu_line_height
                total_price_height = len(price_lines) * price_line_height
                total_text_height = total_menu_height + total_price_height + 5  # 5px 간격
                
                # 텍스트 시작 Y 위치 (박스 중앙 정렬)
                start_y = y + padding + max(0, (text_area_height - total_text_height) // 2)
                
                # 메뉴명 그리기
                current_y = start_y
                for line in menu_lines:
                    try:
                        line_bbox = draw.textbbox((0, 0), line, font=menu_font)
                        line_width = line_bbox[2] - line_bbox[0]
                    except:
                        line_width = len(line) * 10
                    
                    if alignment == 'center':
                        text_x = x + padding + max(0, (text_area_width - line_width) // 2)
                    elif alignment == 'right':
                        text_x = x + width - padding - line_width
                    else:  # left
                        text_x = x + padding
                    
                    draw.text((text_x, current_y), line, fill=(0, 0, 0), font=menu_font)
                    current_y += menu_line_height
                
                # 가격 그리기
                current_y += 5  # 메뉴명과 가격 사이 간격
                for line in price_lines:
                    try:
                        line_bbox = draw.textbbox((0, 0), line, font=price_font)
                        line_width = line_bbox[2] - line_bbox[0]
                    except:
                        line_width = len(line) * 8
                    
                    if alignment == 'center':
                        text_x = x + padding + max(0, (text_area_width - line_width) // 2)
                    elif alignment == 'right':
                        text_x = x + width - padding - line_width
                    else:  # left
                        text_x = x + padding
                    
                    draw.text((text_x, current_y), line, fill=(100, 100, 100), font=price_font)
                    current_y += price_line_height
                
            except Exception as e:
                continue  # 개별 메뉴 처리 실패 시 다음으로 넘어감
        
        return img_copy
        
    except Exception as e:
        st.error(f"텍스트 오버레이 처리 중 오류: {str(e)}")
        return image

def main():
    st.set_page_config(page_title="메뉴보드 생성기", layout="wide")
    
    st.title("🍽️ AI 메뉴보드 생성기")
    st.markdown("Amazon Nova Canvas와 Nova Pro를 사용한 스마트 메뉴보드 생성 도구")
    
    # AWS 클라이언트 초기화
    bedrock_client = init_aws_clients()
    
    # 세션 상태 초기화
    if 'generated_image' not in st.session_state:
        st.session_state.generated_image = None
    if 'text_boxes' not in st.session_state:
        st.session_state.text_boxes = []
    if 'menus' not in st.session_state:
        st.session_state.menus = {}
    if 'image_history' not in st.session_state:
        st.session_state.image_history = []
    if 'current_image_index' not in st.session_state:
        st.session_state.current_image_index = -1
    
    # 사이드바
    with st.sidebar:
        st.header("🎨 이미지 생성 설정")
        
        korean_prompt = st.text_area(
            "메뉴보드 설명 (한국어)",
            value="깔끔하고 모던한 카페 메뉴보드, 여러 메뉴 항목을 위한 빈 공간이 있는 디자인",
            height=100
        )
        
        # 메뉴 정보 유지 옵션
        keep_menu_data = st.checkbox(
            "🔄 기존 메뉴 정보 유지", 
            value=True,
            help="새 이미지 생성 시 기존에 설정한 메뉴 정보를 유지합니다"
        )
        
        if st.button("🌟 메뉴보드 이미지 생성", type="primary"):
            if bedrock_client is None:
                st.error("AWS 클라이언트가 초기화되지 않았습니다.")
            else:
                with st.spinner("이미지를 생성하고 있습니다..."):
                    # 기존 메뉴 정보 백업
                    backup_text_boxes = st.session_state.text_boxes.copy() if keep_menu_data else []
                    backup_menus = st.session_state.menus.copy() if keep_menu_data else {}
                    
                    # 한국어를 영어로 번역
                    english_prompt = translate_to_english(korean_prompt, bedrock_client)
                    st.success(f"번역된 프롬프트: {english_prompt}")
                    
                    # 이미지 생성
                    generated_image = generate_menuboard_image(english_prompt, bedrock_client)
                    if generated_image:
                        # 이미지 히스토리에 추가
                        image_info = {
                            'image': generated_image,
                            'korean_prompt': korean_prompt,
                            'english_prompt': english_prompt,
                            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                            'id': len(st.session_state.image_history)
                        }
                        st.session_state.image_history.append(image_info)
                        st.session_state.current_image_index = len(st.session_state.image_history) - 1
                        st.session_state.generated_image = generated_image
                        
                        if keep_menu_data:
                            # 메뉴 정보 복원
                            st.session_state.text_boxes = backup_text_boxes
                            st.session_state.menus = backup_menus
                            st.success("이미지가 성공적으로 생성되었습니다! (기존 메뉴 정보 유지)")
                        else:
                            # 메뉴 정보 초기화
                            st.session_state.text_boxes = []
                            st.session_state.menus = {}
                            st.success("이미지가 성공적으로 생성되었습니다!")
        
        st.markdown("---")
        
        # 메뉴 정보 저장/불러오기
        st.header("💾 메뉴 정보 관리")
        
        col_save, col_load = st.columns(2)
        
        with col_save:
            if st.button("💾 저장", help="현재 메뉴 구성을 파일로 저장"):
                if st.session_state.text_boxes or st.session_state.menus:
                    menu_data = {
                        'text_boxes': st.session_state.text_boxes,
                        'menus': st.session_state.menus,
                        'timestamp': datetime.now().isoformat()
                    }
                    
                    # JSON으로 저장
                    import json
                    json_data = json.dumps(menu_data, ensure_ascii=False, indent=2)
                    
                    st.download_button(
                        label="📥 메뉴 구성 다운로드",
                        data=json_data,
                        file_name=f"menu_config_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json"
                    )
                else:
                    st.warning("저장할 메뉴 정보가 없습니다.")
        
        with col_load:
            uploaded_file = st.file_uploader(
                "📂 불러오기", 
                type=['json'],
                help="저장된 메뉴 구성 파일을 불러옵니다"
            )
            
            if uploaded_file is not None:
                try:
                    import json
                    menu_data = json.load(uploaded_file)
                    
                    if st.button("✅ 메뉴 구성 적용"):
                        st.session_state.text_boxes = menu_data.get('text_boxes', [])
                        st.session_state.menus = menu_data.get('menus', {})
                        # 키를 정수로 변환 (JSON에서는 문자열로 저장됨)
                        st.session_state.menus = {int(k): v for k, v in st.session_state.menus.items()}
                        st.success("메뉴 구성이 성공적으로 불러와졌습니다!")
                        st.rerun()
                        
                except Exception as e:
                    st.error(f"파일 불러오기 실패: {str(e)}")
        
        st.markdown("---")
        
        # 이미지 갤러리 관리
        st.header("🖼️ 이미지 갤러리")
        
        if st.session_state.image_history:
            st.success(f"총 {len(st.session_state.image_history)}개의 이미지가 저장되어 있습니다.")
            
            # 현재 선택된 이미지 표시
            if st.session_state.current_image_index >= 0:
                current_info = st.session_state.image_history[st.session_state.current_image_index]
                st.info(f"현재 선택: 이미지 #{current_info['id']+1} ({current_info['timestamp']})")
            
            # 이미지 선택 드롭다운
            image_options = []
            for i, img_info in enumerate(st.session_state.image_history):
                option_text = f"이미지 #{img_info['id']+1} - {img_info['timestamp']}"
                if len(img_info['korean_prompt']) > 30:
                    option_text += f" ({img_info['korean_prompt'][:30]}...)"
                else:
                    option_text += f" ({img_info['korean_prompt']})"
                image_options.append(option_text)
            
            selected_index = st.selectbox(
                "이미지 선택",
                range(len(st.session_state.image_history)),
                index=st.session_state.current_image_index if st.session_state.current_image_index >= 0 else 0,
                format_func=lambda x: image_options[x]
            )
            
            # 이미지 선택 버튼들
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button("✅ 선택", help="이 이미지를 현재 작업 이미지로 설정"):
                    st.session_state.current_image_index = selected_index
                    st.session_state.generated_image = st.session_state.image_history[selected_index]['image']
                    st.success("이미지가 선택되었습니다!")
                    st.rerun()
            
            with col2:
                if st.button("🗑️ 삭제", help="선택한 이미지를 갤러리에서 삭제"):
                    if len(st.session_state.image_history) > 1:
                        deleted_info = st.session_state.image_history.pop(selected_index)
                        
                        # 인덱스 재조정
                        if st.session_state.current_image_index == selected_index:
                            if selected_index >= len(st.session_state.image_history):
                                st.session_state.current_image_index = len(st.session_state.image_history) - 1
                            if st.session_state.current_image_index >= 0:
                                st.session_state.generated_image = st.session_state.image_history[st.session_state.current_image_index]['image']
                        elif st.session_state.current_image_index > selected_index:
                            st.session_state.current_image_index -= 1
                        
                        st.success(f"이미지 #{deleted_info['id']+1}가 삭제되었습니다!")
                        st.rerun()
                    else:
                        st.warning("마지막 이미지는 삭제할 수 없습니다.")
            
            with col3:
                if st.button("🧹 전체삭제", help="모든 이미지를 갤러리에서 삭제"):
                    if st.button("⚠️ 확인", key="confirm_delete_all"):
                        st.session_state.image_history = []
                        st.session_state.current_image_index = -1
                        st.session_state.generated_image = None
                        st.success("모든 이미지가 삭제되었습니다!")
                        st.rerun()
            
            # 선택된 이미지 미리보기
            if selected_index < len(st.session_state.image_history):
                selected_info = st.session_state.image_history[selected_index]
                st.image(selected_info['image'], caption=f"이미지 #{selected_info['id']+1}", width=200)
                
                with st.expander("이미지 정보", expanded=False):
                    st.write(f"**생성 시간:** {selected_info['timestamp']}")
                    st.write(f"**한국어 프롬프트:** {selected_info['korean_prompt']}")
                    st.write(f"**영어 프롬프트:** {selected_info['english_prompt']}")
        else:
            st.info("아직 생성된 이미지가 없습니다.")
        
        st.markdown("---")
        
        # 현재 메뉴 정보 상태
        st.header("📊 현재 메뉴 정보")
        
        total_text_boxes = len(st.session_state.text_boxes)
        total_menus = sum(len(menus) for menus in st.session_state.menus.values())
        
        if total_text_boxes > 0 or total_menus > 0:
            st.success(f"📦 텍스트 박스: {total_text_boxes}개")
            st.success(f"🍽️ 메뉴 항목: {total_menus}개")
            
            # 각 텍스트 박스별 메뉴 수 표시
            for i, text_box in enumerate(st.session_state.text_boxes):
                menu_count = len(st.session_state.menus.get(i, []))
                layout = text_box.get('layout', 'grid')
                st.write(f"• 박스 {i+1}: {menu_count}개 메뉴 ({layout}형 배치)")
        else:
            st.info("아직 설정된 메뉴 정보가 없습니다.")
        
        st.markdown("---")
        
        # 메뉴 추가
        st.header("🍽️ 메뉴 추가")
        
        if st.session_state.text_boxes:
            selected_box = st.selectbox(
                "메뉴를 추가할 텍스트 박스 선택",
                range(len(st.session_state.text_boxes)),
                format_func=lambda x: f"텍스트 박스 {x+1}"
            )
            
            with st.form("menu_form"):
                menu_name = st.text_input("메뉴명", placeholder="예: 아메리카노")
                price = st.text_input("가격", placeholder="예: ₩4,500")
                alignment = st.selectbox("정렬", ["left", "center", "right"], index=1)
                
                # 폰트 설정 옵션
                st.subheader("폰트 설정")
                use_auto_font = st.checkbox("자동 폰트 크기 조정", value=True, 
                                          help="체크하면 텍스트가 영역에 맞게 자동으로 폰트 크기가 조정됩니다")
                
                manual_font_size = None
                if not use_auto_font:
                    manual_font_size = st.slider("폰트 크기", min_value=8, max_value=50, value=20)
                
                if st.form_submit_button("메뉴 추가"):
                    if menu_name and price:
                        if selected_box not in st.session_state.menus:
                            st.session_state.menus[selected_box] = []
                        
                        menu_item = {
                            'name': menu_name,
                            'price': price,
                            'alignment': alignment,
                            'use_auto_font': use_auto_font
                        }
                        
                        if not use_auto_font and manual_font_size:
                            menu_item['font_size'] = manual_font_size
                        
                        st.session_state.menus[selected_box].append(menu_item)
                        st.success(f"'{menu_name}' 메뉴가 추가되었습니다!")
                        st.rerun()
            
            # 기존 메뉴 관리
            if selected_box in st.session_state.menus and st.session_state.menus[selected_box]:
                st.subheader("📝 기존 메뉴 관리")
                
                for i, menu in enumerate(st.session_state.menus[selected_box]):
                    with st.expander(f"{menu['name']} - {menu['price']}"):
                        col1, col2 = st.columns([2, 1])
                        
                        with col1:
                            # 폰트 크기 조정
                            current_auto = menu.get('use_auto_font', True)
                            new_auto = st.checkbox(f"자동 폰트 조정", value=current_auto, key=f"auto_{selected_box}_{i}")
                            
                            if not new_auto:
                                current_size = menu.get('font_size', 20)
                                new_size = st.slider(f"폰트 크기", min_value=8, max_value=50, 
                                                   value=current_size, key=f"size_{selected_box}_{i}")
                                
                                if st.button(f"폰트 크기 적용", key=f"apply_{selected_box}_{i}"):
                                    st.session_state.menus[selected_box][i]['use_auto_font'] = new_auto
                                    st.session_state.menus[selected_box][i]['font_size'] = new_size
                                    st.success("폰트 크기가 적용되었습니다!")
                                    st.rerun()
                            else:
                                if st.button(f"자동 조정 적용", key=f"auto_apply_{selected_box}_{i}"):
                                    st.session_state.menus[selected_box][i]['use_auto_font'] = True
                                    if 'font_size' in st.session_state.menus[selected_box][i]:
                                        del st.session_state.menus[selected_box][i]['font_size']
                                    st.success("자동 폰트 조정이 적용되었습니다!")
                                    st.rerun()
                        
                        with col2:
                            if st.button(f"삭제", key=f"delete_{selected_box}_{i}", type="secondary"):
                                st.session_state.menus[selected_box].pop(i)
                                st.success("메뉴가 삭제되었습니다!")
                                st.rerun()
            
            # 레이아웃 설정
            if selected_box in st.session_state.menus and st.session_state.menus[selected_box]:
                st.subheader("📐 레이아웃 설정")
                layout_type = st.selectbox(
                    "배치 방식",
                    ["grid", "vertical", "horizontal"],
                    format_func=lambda x: {"grid": "격자형", "vertical": "세로형", "horizontal": "가로형"}[x]
                )
                
                if st.button("레이아웃 적용"):
                    st.session_state.text_boxes[selected_box]['layout'] = layout_type
                    st.rerun()
        else:
            st.info("먼저 텍스트 박스를 추가해주세요.")
    
    # 메인 영역
    if st.session_state.generated_image is not None:
        # 탭으로 구성
        tab1, tab2 = st.tabs(["🎨 메뉴보드 작업", "🖼️ 이미지 갤러리"])
        
        with tab1:
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.subheader("영역 선택")
                
                # 캔버스 크기 설정
                original_width = st.session_state.generated_image.width
                original_height = st.session_state.generated_image.height
                canvas_width = 500
                canvas_height = int(canvas_width * original_height / original_width)
                
                canvas_result = st_canvas(
                    fill_color="rgba(255, 0, 0, 0.3)",
                    stroke_width=2,
                    stroke_color="#FF0000",
                    background_image=st.session_state.generated_image,
                    height=canvas_height,
                    width=canvas_width,
                    drawing_mode="rect",
                    key="canvas"
                )
                
                # 선택된 영역을 텍스트 박스로 추가
                if canvas_result.json_data is not None:
                    objects = canvas_result.json_data["objects"]
                    if objects:
                        last_rect = objects[-1]
                        if last_rect["type"] == "rect":
                            scale_x = original_width / canvas_width
                            scale_y = original_height / canvas_height
                            
                            real_x = int(last_rect["left"] * scale_x)
                            real_y = int(last_rect["top"] * scale_y)
                            real_width = int(last_rect["width"] * scale_x)
                            real_height = int(last_rect["height"] * scale_y)
                            
                            st.info(f"선택된 영역: X={real_x}, Y={real_y}, 너비={real_width}, 높이={real_height}")
                            
                            if st.button("이 영역을 텍스트 박스로 추가", type="primary"):
                                new_text_box = {
                                    'bbox': (real_x, real_y, real_width, real_height),
                                    'layout': 'grid'
                                }
                                st.session_state.text_boxes.append(new_text_box)
                                st.success(f"텍스트 박스 {len(st.session_state.text_boxes)}가 추가되었습니다!")
                                st.rerun()
            
            with col2:
                st.subheader("완성된 메뉴보드")
                
                # 전체 폰트 설정
                with st.expander("🔧 전체 폰트 설정", expanded=False):
                    st.info("개별 메뉴 설정이 우선 적용됩니다.")
                    
                    global_auto_font = st.checkbox("전체 자동 폰트 조정", value=True, 
                                                 help="모든 메뉴에 자동 폰트 조정을 적용합니다 (개별 설정이 우선)")
                    
                    global_font_size = None
                    if not global_auto_font:
                        global_font_size = st.slider("전체 기본 폰트 크기", min_value=8, max_value=50, value=20)
                    
                    if st.button("전체 설정 적용"):
                        # 모든 메뉴에 전체 설정 적용 (개별 설정이 없는 경우만)
                        for box_idx in st.session_state.menus:
                            for menu_idx, menu in enumerate(st.session_state.menus[box_idx]):
                                # 개별 설정이 없는 메뉴만 업데이트
                                if 'use_auto_font' not in menu:
                                    st.session_state.menus[box_idx][menu_idx]['use_auto_font'] = global_auto_font
                                    if not global_auto_font and global_font_size:
                                        st.session_state.menus[box_idx][menu_idx]['font_size'] = global_font_size
                        st.success("전체 폰트 설정이 적용되었습니다!")
                        st.rerun()
                
                # 이미지 크기 조절
                image_width = st.slider("미리보기 크기", 400, 1000, 600)
                
                # 모든 메뉴 위치 계산
                all_menu_positions = []
                for i, text_box in enumerate(st.session_state.text_boxes):
                    if i in st.session_state.menus and st.session_state.menus[i]:
                        layout_type = text_box.get('layout', 'grid')
                        positions = calculate_menu_positions(
                            text_box['bbox'], 
                            st.session_state.menus[i], 
                            layout_type
                        )
                        all_menu_positions.extend(positions)
                
                if all_menu_positions:
                    # 실시간 미리보기
                    final_image = add_text_overlay(st.session_state.generated_image, all_menu_positions)
                    st.image(final_image, caption="완성된 메뉴보드", width=image_width)
                    
                    # 메뉴 정보 표시
                    with st.expander("📋 메뉴 정보", expanded=False):
                        for i, position in enumerate(all_menu_positions):
                            font_info = "자동 조정" if position.get('use_auto_font', True) else f"크기 {position.get('font_size', '기본')}"
                            st.write(f"**{position['menu_name']}** - {position['price']} (폰트: {font_info})")
                    
                    # 다운로드 버튼
                    img_buffer = io.BytesIO()
                    final_image.save(img_buffer, format='PNG')
                    img_buffer.seek(0)
                    
                    st.download_button(
                        label="📥 완성된 메뉴보드 다운로드",
                        data=img_buffer.getvalue(),
                        file_name="menuboard.png",
                        mime="image/png",
                        type="primary"
                    )
                    
                    # 추가 옵션
                    col_a, col_b = st.columns(2)
                    with col_a:
                        if st.button("🔄 모든 메뉴 초기화", type="secondary"):
                            st.session_state.menus = {}
                            st.success("모든 메뉴가 초기화되었습니다!")
                            st.rerun()
                    
                    with col_b:
                        if st.button("📦 텍스트 박스 초기화", type="secondary"):
                            st.session_state.text_boxes = []
                            st.session_state.menus = {}
                            st.success("텍스트 박스가 초기화되었습니다!")
                            st.rerun()
                else:
                    st.image(st.session_state.generated_image, caption="원본 이미지", width=image_width)
                    st.info("텍스트 박스를 추가하고 메뉴를 입력해주세요.")
        
        with tab2:
            st.subheader("🖼️ 이미지 갤러리")
            
            if st.session_state.image_history:
                # 갤러리 그리드 표시
                cols_per_row = 3
                for i in range(0, len(st.session_state.image_history), cols_per_row):
                    cols = st.columns(cols_per_row)
                    
                    for j in range(cols_per_row):
                        idx = i + j
                        if idx < len(st.session_state.image_history):
                            img_info = st.session_state.image_history[idx]
                            
                            with cols[j]:
                                # 현재 선택된 이미지 표시
                                is_current = (idx == st.session_state.current_image_index)
                                border_color = "🟢" if is_current else "⚪"
                                
                                st.image(img_info['image'], 
                                        caption=f"{border_color} 이미지 #{img_info['id']+1}",
                                        width=200)
                                
                                st.write(f"**생성시간:** {img_info['timestamp']}")
                                
                                # 프롬프트 표시 (짧게)
                                short_prompt = img_info['korean_prompt'][:50] + "..." if len(img_info['korean_prompt']) > 50 else img_info['korean_prompt']
                                st.write(f"**프롬프트:** {short_prompt}")
                                
                                # 버튼들
                                col_btn1, col_btn2 = st.columns(2)
                                
                                with col_btn1:
                                    if st.button("✅ 선택", key=f"select_{idx}"):
                                        st.session_state.current_image_index = idx
                                        st.session_state.generated_image = img_info['image']
                                        st.success(f"이미지 #{img_info['id']+1}이 선택되었습니다!")
                                        st.rerun()
                                
                                with col_btn2:
                                    if st.button("🗑️ 삭제", key=f"delete_{idx}"):
                                        if len(st.session_state.image_history) > 1:
                                            st.session_state.image_history.pop(idx)
                                            
                                            # 인덱스 재조정
                                            if st.session_state.current_image_index == idx:
                                                if idx >= len(st.session_state.image_history):
                                                    st.session_state.current_image_index = len(st.session_state.image_history) - 1
                                                if st.session_state.current_image_index >= 0:
                                                    st.session_state.generated_image = st.session_state.image_history[st.session_state.current_image_index]['image']
                                            elif st.session_state.current_image_index > idx:
                                                st.session_state.current_image_index -= 1
                                            
                                            st.success("이미지가 삭제되었습니다!")
                                            st.rerun()
                                        else:
                                            st.warning("마지막 이미지는 삭제할 수 없습니다.")
                                
                                # 상세 정보 확장
                                with st.expander("상세 정보"):
                                    st.write(f"**한국어 프롬프트:** {img_info['korean_prompt']}")
                                    st.write(f"**영어 프롬프트:** {img_info['english_prompt']}")
                                    
                                    # 개별 이미지 다운로드
                                    img_buffer = io.BytesIO()
                                    img_info['image'].save(img_buffer, format='PNG')
                                    img_buffer.seek(0)
                                    
                                    st.download_button(
                                        label="📥 이미지 다운로드",
                                        data=img_buffer.getvalue(),
                                        file_name=f"menuboard_{img_info['id']+1}.png",
                                        mime="image/png",
                                        key=f"download_{idx}"
                                    )
                
                # 전체 관리 버튼
                st.markdown("---")
                col_manage1, col_manage2 = st.columns(2)
                
                with col_manage1:
                    if st.button("🧹 전체 삭제", type="secondary"):
                        if st.button("⚠️ 정말 삭제하시겠습니까?", key="confirm_delete_all_main"):
                            st.session_state.image_history = []
                            st.session_state.current_image_index = -1
                            st.session_state.generated_image = None
                            st.success("모든 이미지가 삭제되었습니다!")
                            st.rerun()
                
                with col_manage2:
                    # 모든 이미지 일괄 다운로드 (ZIP)
                    if st.button("📦 전체 다운로드 (ZIP)"):
                        import zipfile
                        
                        zip_buffer = io.BytesIO()
                        with zipfile.ZipFile(zip_buffer, 'w') as zip_file:
                            for i, img_info in enumerate(st.session_state.image_history):
                                img_buffer = io.BytesIO()
                                img_info['image'].save(img_buffer, format='PNG')
                                img_buffer.seek(0)
                                
                                zip_file.writestr(
                                    f"menuboard_{img_info['id']+1}_{img_info['timestamp'].replace(':', '-')}.png",
                                    img_buffer.getvalue()
                                )
                        
                        zip_buffer.seek(0)
                        
                        st.download_button(
                            label="📥 ZIP 파일 다운로드",
                            data=zip_buffer.getvalue(),
                            file_name=f"menuboard_gallery_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip",
                            mime="application/zip"
                        )
            else:
                st.info("아직 생성된 이미지가 없습니다.")
    
    else:
        st.info("👈 사이드바에서 메뉴보드 이미지를 먼저 생성해주세요.")
        
        # 메뉴 정보 유지 기능 안내
        with st.expander("💡 새로운 기능 안내", expanded=True):
            st.markdown("""
            ### 🔄 메뉴 정보 유지 기능
            - **기존 메뉴 정보 유지** 체크박스를 활성화하면 새로운 배경 이미지를 생성할 때 기존에 설정한 메뉴 정보가 그대로 유지됩니다.
            - 배경만 바꾸고 싶을 때 매우 유용합니다!
            
            ### 🖼️ 이미지 갤러리 기능
            - 생성된 모든 이미지가 자동으로 갤러리에 저장됩니다.
            - **이미지 갤러리** 탭에서 모든 이미지를 한눈에 비교할 수 있습니다.
            - 원하는 이미지를 선택하여 작업 이미지로 설정할 수 있습니다.
            - 개별 이미지 삭제 또는 전체 이미지 일괄 다운로드 가능합니다.
            
            ### 💾 메뉴 정보 저장/불러오기
            - 완성한 메뉴 구성을 JSON 파일로 저장할 수 있습니다.
            - 저장된 파일을 불러와서 다른 배경 이미지에 동일한 메뉴 구성을 적용할 수 있습니다.
            
            ### 🎨 개선된 폰트 기능
            - **자동 폰트 조정**: 텍스트가 영역에 맞게 자동으로 크기 조절
            - **수동 폰트 설정**: 원하는 폰트 크기를 직접 지정
            - **개별 메뉴 설정**: 각 메뉴마다 다른 폰트 설정 가능
            """)
    
    # 사용 팁
    with st.sidebar:
        with st.expander("💡 사용 팁", expanded=False):
            st.markdown("""
            **효율적인 작업 순서:**
            1. 첫 번째 배경 이미지 생성
            2. 텍스트 박스 추가 및 메뉴 입력
            3. 메뉴 구성 저장 (백업용)
            4. '기존 메뉴 정보 유지' 체크
            5. 다른 배경 이미지 생성
            6. 이미지 갤러리에서 원하는 배경 선택
            7. 동일한 메뉴 구성으로 다양한 배경 테스트!
            
            **이미지 갤러리 활용:**
            • 생성된 모든 이미지가 자동으로 저장됩니다
            • 갤러리 탭에서 이미지를 비교하고 선택할 수 있습니다
            • 개별 또는 전체 이미지 다운로드 가능
            • 불필요한 이미지는 언제든 삭제 가능
            """)

if __name__ == "__main__":
    main()
