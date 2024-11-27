import json
import os
from typing import Dict, Any

from dotenv import load_dotenv

load_dotenv()

class ConfigLoader:
    def __init__(self, config_path: str = "configuration/config.json"):
        self.config_path = config_path
        self.config = self._load_config()
        
        # Load Azure config from environment variables
        self.azure_config = {
            'api_key': os.getenv('OPENAI_API_KEY'),
            'api_version': "2024-02-15-preview",
            'azure_endpoint': os.getenv('OPENAI_ENDPOINT'),
            'deployment_name': os.getenv('OPENAI_DEPLOYMENT_NAME')
        }
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from JSON file"""
        try:
            with open(self.config_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Configuration file not found at {self.config_path}")
        except json.JSONDecodeError:
            raise ValueError(f"Invalid JSON in configuration file {self.config_path}")
    
    def save_config(self) -> None:
        """Save current configuration to JSON file"""
        with open(self.config_path, 'w') as f:
            json.dump(self.config, f, indent=4)
    
    def get_agent_config(self, agent_name: str) -> Dict[str, Any]:
        """Get configuration for a specific agent"""
        return self.config.get(agent_name, {})
    
    def get_processing_config(self) -> Dict[str, Any]:
        """Get document processing configuration"""
        return self.config.get('document_processing', {})
    
    def get_azure_config(self) -> Dict[str, Any]:
        """Get Azure configuration"""
        return self.azure_config
    
    def update_config(self, section: str, key: str, value: Any) -> bool:
        """Update a specific configuration value"""
        if section in self.config and key in self.config[section]:
            self.config[section][key] = value
            self.save_config()
            return True
        return False
    
