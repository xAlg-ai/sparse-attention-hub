"""Result storage for benchmark results."""

import json
import os
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional


class ResultStorage:
    """Manages storage and retrieval of benchmark results."""

    def __init__(self, storage_path: str = "./benchmark_results"):
        self.storage_path = storage_path
        self._ensure_storage_directory()

    def _ensure_storage_directory(self) -> None:
        """Ensure the storage directory exists."""
        os.makedirs(self.storage_path, exist_ok=True)

    def store(self, results: List[str]) -> str:
        """Store benchmark results.

        Args:
            results: List of result strings to store

        Returns:
            Unique identifier for the stored results
        """
        result_id = str(uuid.uuid4())
        timestamp = datetime.now().isoformat()

        result_data = {"id": result_id, "timestamp": timestamp, "results": results}

        file_path = os.path.join(self.storage_path, f"{result_id}.json")

        with open(file_path, "w") as f:
            json.dump(result_data, f, indent=2)

        return result_id

    def load(self, result_id: str) -> Optional[Dict[str, Any]]:
        """Load benchmark results by ID.

        Args:
            result_id: Unique identifier for the results

        Returns:
            Result data if found, None otherwise
        """
        file_path = os.path.join(self.storage_path, f"{result_id}.json")

        if not os.path.exists(file_path):
            return None

        with open(file_path, "r") as f:
            result = json.load(f)
            return result

    def list_results(self) -> List[Dict[str, str]]:
        """List all stored results.

        Returns:
            List of result metadata (id, timestamp)
        """
        results = []

        for filename in os.listdir(self.storage_path):
            if filename.endswith(".json"):
                result_id = filename[:-5]  # Remove .json extension
                result_data = self.load(result_id)

                if result_data:
                    results.append(
                        {"id": result_data["id"], "timestamp": result_data["timestamp"]}
                    )

        return sorted(results, key=lambda x: x["timestamp"], reverse=True)

    def delete(self, result_id: str) -> bool:
        """Delete stored results.

        Args:
            result_id: Unique identifier for the results to delete

        Returns:
            True if deleted successfully, False if not found
        """
        file_path = os.path.join(self.storage_path, f"{result_id}.json")

        if os.path.exists(file_path):
            os.remove(file_path)
            return True

        return False

    def clear_all(self) -> int:
        """Clear all stored results.

        Returns:
            Number of results deleted
        """
        count = 0

        for filename in os.listdir(self.storage_path):
            if filename.endswith(".json"):
                file_path = os.path.join(self.storage_path, filename)
                os.remove(file_path)
                count += 1

        return count
