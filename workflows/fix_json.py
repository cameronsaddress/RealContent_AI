
import re
import json

file_path = "/home/canderson/n8n/workflows/COMPLETE_PIPELINE.json"

try:
    with open(file_path, "r") as f:
        content = f.read()

    # 1. Identify the corrupt block start
    # It started around "Process & Save Trends"
    # The corruption is that "Schedule Trigger" and "Webhook" specific keys became nested/malformed.
    
    # We will look for "Process & Save Trends": { ... } and the "Schedule Trigger" inside it.
    # We want to replace the MESS with clean connections.
    
    # Define the start pattern (Process & Save Trends key)
    start_pattern = r'(\s+)"Process & Save Trends": \{(?:.|\n)*?"Respond to Webhook"(?:.|\n)*?"Schedule Trigger": \{(?:.|\n)*?"Webhook": \{'
    
    # We can't match potentially infinite nesting easily with regex.
    # Instead, let's find the start of connections object and rebuild it? 
    # No, that's hard.
    
    # Improved strategy:
    # 1. Capture the text BEFORE "Process & Save Trends".
    # 2. Capture the text AFTER the last node in the flow ("Serve Video"? "Read Video"?).
    # 3. Reconstruct the middle (Connections) cleanly?
    # No, there are too many nodes.
    
    # Let's fix the specific syntax errors first.
    # The error is ` "Webhook": {` appearing inside an array/object incorrectly.
    # And excessive indentation.
    
    # Step A: Fix the "Process & Save Trends" closing.
    # It currently ends with `... "Respond to Webhook" ... } ] ] "Schedule Trigger": ...` which is wrong (missing close brackets).
    # And "Schedule Trigger" is indented too far?
    
    # Let's just find the "Process & Save Trends" block and force-close it, then start "Schedule Trigger".
    
    # Find the line with "Process & Save Trends": {
    # Then find the line with "Schedule Trigger": {
    # Replace everything in between with the correct Process & Save trends body.
    
    new_process_save = """    "Process & Save Trends": {
      "main": [
        [
          {
            "node": "Respond to Webhook",
            "type": "main",
            "index": 0
          }
        ]
      ]
    },"""
    
    # Regex to find the bad block
    # Matches: "Process & Save Trends": { ...anything... "Schedule Trigger": {
    bad_block_regex = r'(\s+)"Process & Save Trends": \{(?:.|\n)*?"node": "Respond to Webhook",(?:.|\n)*?(\s+)"Schedule Trigger": \{'
    
    # We replace it with new_process_save + \n    "Schedule Trigger": {
    
    match = re.search(bad_block_regex, content)
    if not match:
        print("Could not find the specific corrupt block pattern.")
        # Fallback: Print what we see around there
        idx = content.find('"Process & Save Trends": {')
        if idx != -1:
            print("Found start at", idx)
            print(content[idx:idx+500])
        exit(1)
        
    print("Found corrupt block. Fixing structure...")
    
    # This replacement fixes the transition from Process&Save to Schedule Trigger.
    # Now valid: "Process ...": { ... }, "Schedule Trigger": {
    
    # But "Schedule Trigger" contained "Webhook" inside its array in the bad version.
    # bad: "Schedule Trigger": { "main": [ [ "Webhook": { ...
    # We need to fix that too.
    
    # Let's construct the first few correct connection blocks manually.
    
    correct_header = """    "Process & Save Trends": {
      "main": [
        [
          {
            "node": "Respond to Webhook",
            "type": "main",
            "index": 0
          }
        ]
      ]
    },
    "Schedule Trigger": {
      "main": [
        [
          {
            "node": "Get Approved Idea",
            "type": "main",
            "index": 0
          }
        ]
      ]
    },
    "Webhook": {
      "main": [
        [
          {
            "node": "Respond Immediately",
            "type": "main",
            "index": 0
          }
        ]
      ]
    },
    "Respond Immediately": {
      "main": [
        [
          {
            "node": "Get Approved Idea",
            "type": "main",
            "index": 0
          }
        ]
      ]
    },
    "Get Approved Idea": {"""
    
    # We want to replace from "Process & Save Trends" start ... down to "Get Approved Idea" start.
    # The bad file has:
    # "Process ...": ... "Schedule Trigger" ... "Webhook" ... "Respond Immediately" ... "Get Approved Idea" ...
    # All nested.
    
    # Regex: replace everything from "Process & Save Trends": { ... up to ... "Get Approved Idea": {
    # WITH correct_header.
    
    pattern = r'(\s+)"Process & Save Trends": \{(?:.|\n)*?"Get Approved Idea": \{'
    content_fixed = re.sub(pattern, correct_header, content, count=1)
    
    # Now we have fixed the top level. But "Get Approved Idea" body in the file might still be indented/malformed?
    # No, "Get Approved Idea": { was the anchor. The content *after* it is preserved.
    # But wait, in the bad file, "Get Approved Idea" was nested deep.
    # So the *closing* braces for the previous nesting (which we removed) are missing?
    # No, we removed the opening braces of the bad nesting.
    # But the closing braces at the END of the connections object might be excessive now?
    # Or insufficient?
    
    # Also, the keys AFTER "Get Approved Idea" (like "Specific ID?") are indented deep (18 spaces).
    # We need to un-indent them.
    # They look like: \n                  "Specific ID?": {
    # We want: \n    "Specific ID?": {
    
    # Regex to reducing spacing for keys.
    # Matches \n followed by >4 spaces followed by "Key": {
    # We replace with \n    "Key": {
    
    def unindent_keys(match):
        key = match.group(2)
        return f'\n    "{key}": {{'
        
    # Pattern: newline, then 6+ spaces, then "Key": {
    content_fixed = re.sub(r'\n(\s{6,})"([^"]+)": \{', unindent_keys, content_fixed)
    
    # Finally, check for trailing closing braces.
    # The original file had deep nesting, so it likely had many }}}}} at the end of "connections".
    # We removed the OPENING of that nesting.
    # So now we have too many closing braces at the end?
    # We should parse the JSON to verify.
    # If invalid, we can count braces or try to trim.
    
    try:
        json.loads(content_fixed)
        print("JSON is valid!")
    except json.JSONDecodeError as e:
        print(f"JSON invalid after fix: {e}")
        # Identify where.
        # If it says "Extra data", we have too many closing braces.
        lines = content_fixed.split('\n')
        # Simple heuristic: The "connections" object ends usually before "settings".
        # Let's inspect around "settings".
        
        # Regex for connections closing:
        # It usually looks like:
        #       }
        #     },
        #     "settings": {
        
        # If we have extra braces:
        #                   }
        #                 ]
        #               ]
        #             }
        #           ]
        #         ]
        #       }
        #     }, "settings" ...
        
        # We can try to normalize the closing braces block before "settings".
        # Find "settings": {
        # Check text before it.
        pass

    with open(file_path, "w") as f:
        f.write(content_fixed)
    print("File saved.")

except Exception as e:
    print(f"Error: {e}")
