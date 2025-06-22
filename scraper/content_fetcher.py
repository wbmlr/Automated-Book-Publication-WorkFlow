from playwright.sync_api import sync_playwright, Error

def fetch_content_and_screenshot(url: str):
    print(f"--- Starting new scrape for: {url} ---")
    with sync_playwright() as p:
        browser = p.chromium.launch()
        try:
            page = browser.new_page()
            page.goto(url, wait_until="domcontentloaded", timeout=60000)
            
            # Get screenshot as bytes
            screenshot_bytes = page.screenshot()
            
            # Get content text
            content_element = page.locator(".mw-parser-output")
            content_text = content_element.inner_text()
            
            return {"text": content_text, "screenshot_bytes": screenshot_bytes}
        except Error as e:
            print(f"❌ Scraping Error: {e}")
            return None
        finally:
            browser.close()