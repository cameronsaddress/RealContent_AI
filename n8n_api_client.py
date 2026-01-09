import requests
import json
import os

class N8NClient:
    def __init__(self, base_url="http://localhost:5678/api/v1", api_key=None):
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key or os.getenv("N8N_API_KEY")
        
        if not self.api_key:
            raise ValueError("API Key is required. Pass it to init or set N8N_API_KEY env var.")
            
        self.headers = {
            "X-N8N-API-KEY": self.api_key,
            "Content-Type": "application/json"
        }

    def _handle_response(self, response):
        try:
            response.raise_for_status()
            return response.json()
        except requests.exceptions.HTTPError as e:
            print(f"API Error: {e}")
            print(f"Response: {response.text}")
            raise

    def get_workflows(self):
        """List all workflows."""
        url = f"{self.base_url}/workflows"
        resp = requests.get(url, headers=self.headers)
        return self._handle_response(resp)

    def get_workflow(self, workflow_id):
        """Get details of a specific workflow."""
        url = f"{self.base_url}/workflows/{workflow_id}"
        resp = requests.get(url, headers=self.headers)
        return self._handle_response(resp)

    def update_workflow(self, workflow_id, workflow_data):
        """Update a workflow with new JSON definition."""
        url = f"{self.base_url}/workflows/{workflow_id}"
        # n8n API expects { "nodes": [...], "connections": {...}, ... }
        # wrapper object might be needed depending on endpoint, usually PUT /workflows/:id takes the workflow object directly.
        resp = requests.put(url, headers=self.headers, json=workflow_data)
        return self._handle_response(resp)

    def activate_workflow(self, workflow_id):
        """Activate a workflow."""
        url = f"{self.base_url}/workflows/{workflow_id}/activate"
        resp = requests.post(url, headers=self.headers)
        return self._handle_response(resp)

    def deactivate_workflow(self, workflow_id):
        """Deactivate a workflow."""
        url = f"{self.base_url}/workflows/{workflow_id}/deactivate"
        resp = requests.post(url, headers=self.headers)
        return self._handle_response(resp)

    def import_workflow_from_file(self, workflow_id, file_path):
        """Reads a local JSON file and updates the workflow in n8n."""
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        # If the file contains a wrapper (like an export array), extract the workflow
        if isinstance(data, list):
            data = data[0] # Assume single workflow export
            
        return self.update_workflow(workflow_id, data)

# Example Usage:
# client = N8NClient(api_key="your_key_here")
# client.import_workflow_from_file("LNpBI12tR5p3NX3g", "workflows/COMPLETE_PIPELINE.json")
