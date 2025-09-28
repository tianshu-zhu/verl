"""
Pod Manager for SWE Training Framework.

This module provides Kubernetes pod management capabilities for running software
engineering agents in isolated execution environments. It handles pod lifecycle
management, command execution, and resource cleanup for distributed training workflows.

Key Components:
- PodManager: Main class for managing Kubernetes pods
- Pod creation and deletion with proper resource management
- Command execution within pods with timeout handling
- Environment setup and tool integration
- Performance monitoring and logging

The pod manager integrates with the kodo package for container orchestration and
provides a clean interface for agent execution engines to interact with isolated
execution environments in Kubernetes clusters.

Features:
- Asynchronous pod operations for improved performance
- Automatic resource cleanup and error handling
- Tool integration for software engineering tasks
- Configurable timeouts and retry mechanisms
- Comprehensive logging for debugging and monitoring

Usage:
    pod_manager = PodManager(config)
    pod_name = await pod_manager.create_pod(image="ubuntu:20.04")
    result = await pod_manager.execute_command(pod_name, "echo 'Hello World'")
    await pod_manager.kill_pod(pod_name)
"""

import asyncio
import os
import sys
import time
import uuid
import logging
import importlib.util
from typing import Dict, Any, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)

# Import kodo for pod management
try:
    from kodo import ContainerRunner
except ImportError:
    print("ERROR: kodo package not found. Please install it with: pip install kodo")
    ContainerRunner = None
    
# 尝试导入workers包并获取其路径
try:
    import workers
    R2EGYM_PATH = os.path.dirname(workers.__file__)
except ImportError:
    # 如果workers包不存在，报错
    raise ImportError("workers package not found")

# List of tools to be used in the environment.
R2EGYM_COMMAND_FILES = [
    os.path.join(R2EGYM_PATH, "tools/r2e_tools_offical/file_editor.py"),
    os.path.join(R2EGYM_PATH, "tools/r2e_tools_offical/search.py"),
    os.path.join(R2EGYM_PATH, "tools/r2e_tools_offical/execute_bash.py"),
    os.path.join(R2EGYM_PATH, "tools/r2e_tools_offical/finish.py"),
]


class PodManager:
    """
    Manages Kubernetes pods for agent execution environments.
    
    Supports creating, killing, resetting pods and executing commands within them.
    """
    
    def __init__(self, config, max_workers=64):
        """
        Initialize PodManager with configuration.
        
        Args:
            config: Configuration object containing pod settings
            max_workers: Maximum number of worker threads for concurrent operations
        """
        self.config = config
        self.namespace = config.agent.namespace
        self.kubeconfig_path = config.agent.kubeconfig_path
        self.working_dir = config.agent.working_dir or "/testbed"
        
        # Thread pool for async operations
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        
        # Pod registry to track active pods
        self.active_pods = {}  # {pod_name: pod_info}
        
        # Initialize ContainerRunner
        if ContainerRunner:
            self.kodo_runner = ContainerRunner(
                backend="kubernetes",
                namespace=self.namespace,
                kubeconfig_path=self.kubeconfig_path
            )
        else:
            raise ImportError("kodo package is required for pod management")
    
    def create_pod(self, env_args: Dict[str, Any], pod_prefix: str = "deep-trainer") -> Tuple[str, Dict[str, Any]]:
        """
        Create a new pod with given environment arguments.
        
        Args:
            env_args: Environment arguments containing docker image and other settings
            pod_prefix: Prefix for pod name generation
            
        Returns:
            Tuple of (pod_name, pod_info) where pod_info contains creation metadata
        """
        max_retries = 3
        retry_delay = 5
        pod_name = f"{pod_prefix}-{str(uuid.uuid4())}"
        
        # Extract docker image from env_args
        if "docker_image" in env_args:
            image = env_args["docker_image"]
        elif "docker" in env_args:
            image = env_args["docker"]
        else:
            raise ValueError(f"docker_image or docker not found in env_args: {env_args}")
        
        # Prepare Kubernetes configuration
        k8s_config = {
            "execution_mode": "k8s",
            "pod_name": pod_name,
            "namespace": self.namespace,
            "kubeconfig_path": self.kubeconfig_path,
            "working_dir": self.working_dir
        }
        
        # Environment variables for the container
        environment = {
            "PYTHONPATH": "/testbed",
            # UTF-8 encoding settings for LLM-generated code
            # This prevents UnicodeEncodeError when LLM generates Unicode characters
            # like checkmarks (✓), crosses (✗), or other special symbols
            "PYTHONIOENCODING": "utf-8",
            "LANG": "C.UTF-8",
            "LC_ALL": "C.UTF-8",
            # Proxy settings
            "http_proxy": "http://agent.baidu.com:8891",
            "https_proxy": "http://agent.baidu.com:8891",
            "PIP_INDEX_URL": "http://pip.baidu.com/pypi/simple",
            "PIP_TRUSTED_HOST": "pip.baidu.com"
        }
        
        # Try to start pod up to max_retries
        for attempt in range(max_retries):
            try:
                print(f"[PodManager] Creating pod {pod_name} (attempt {attempt+1}/{max_retries})")
                print(f"[PodManager] Request parameters - image: {image}, name: {pod_name}, environment: {environment}")
                self.kodo_runner.start_container(
                    image,
                    name=pod_name,
                    environment=environment,
                    node_selector={"nvme": "ok"}
                )
                
                # Store pod info
                pod_info = {
                    "pod_name": pod_name,
                    "image": image,
                    "k8s_config": k8s_config,
                    "env_args": env_args,
                    "created_at": time.time(),
                    "status": "running"
                }
                self.active_pods[pod_name] = pod_info
                
                print(f"[PodManager] Successfully created pod {pod_name}")
                return pod_name, pod_info
                
            except Exception as e:
                if attempt < max_retries - 1:
                    print(f"[PodManager] Failed to create pod {pod_name} (attempt {attempt+1}/{max_retries}): {e}")
                    print(f"[PodManager] Failed request parameters - image: {image}, name: {pod_name}, environment: {environment}")
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                else:
                    print(f"[PodManager] Failed to create pod {pod_name} after {max_retries} attempts: {e}")
                    print(f"[PodManager] Final failed request parameters - image: {image}, name: {pod_name}, environment: {environment}")
                    raise
    
    def kill_pod(self, pod_name: str, force: bool = True) -> bool:
        """
        Kill/delete a pod.
        
        Args:
            pod_name: Name of the pod to kill
            force: Whether to force delete the pod
            
        Returns:
            bool: True if successfully killed, False otherwise
        """
        try:
            print(f"[PodManager] Killing pod {pod_name}")
            
            # Stop the container using kodo
            self.kodo_runner.stop_container(pod_name)
            
            # Remove from active pods registry
            if pod_name in self.active_pods:
                self.active_pods[pod_name]["status"] = "killed"
                del self.active_pods[pod_name]
            
            print(f"[PodManager] Successfully killed pod {pod_name}")
            return True
            
        except Exception as e:
            print(f"[PodManager] Error killing pod {pod_name}: {e}")
            return False
    
    def reset_pod(self, old_pod_name: str, env_args: Dict[str, Any], pod_prefix: str = "deep-trainer") -> Tuple[str, Dict[str, Any]]:
        """
        Reset a pod by killing the old one and creating a new one with different name.
        
        Args:
            old_pod_name: Name of the pod to reset
            env_args: Environment arguments for the new pod
            pod_prefix: Prefix for new pod name generation
            
        Returns:
            Tuple of (new_pod_name, new_pod_info)
        """
        print(f"[PodManager] Resetting pod {old_pod_name}")
        
        # Kill the old pod
        self.kill_pod(old_pod_name)
        
        # Create a new pod with different name
        new_pod_name, new_pod_info = self.create_pod(env_args, pod_prefix)
        
        print(f"[PodManager] Pod reset completed: {old_pod_name} -> {new_pod_name}")
        return new_pod_name, new_pod_info
    
    def execute_command(self, pod_name: str, command: str, timeout: int = 300) -> Tuple[str, int]:
        """
        Execute a command in the specified pod.
        
        Args:
            pod_name: Name of the pod to execute command in
            command: Command to execute
            timeout: Command timeout in seconds
            
        Returns:
            Tuple of (output, exit_code)
        """
        try:
            if pod_name not in self.active_pods:
                print(f"[PodManager] Warning: Pod {pod_name} not found in active pods registry")
            
            start_time = time.time()
            output, exit_code = self.kodo_runner.execute_command(pod_name, command)
            end_time = time.time()
            elapsed = end_time - start_time
            print(f"[PodManager] Command '{command}' executed in pod {pod_name} in {elapsed:.3f} seconds")
            return output, exit_code
            
        except Exception as e:
            print(f"[PodManager] Error executing command '{command}' in pod {pod_name}: {e}")
            return str(e), -1
    
    async def execute_command_async(self, pod_name: str, command: str, timeout: int = 300) -> Tuple[str, int]:
        """
        Execute a command in the specified pod asynchronously.
        
        Args:
            pod_name: Name of the pod to execute command in
            command: Command to execute
            timeout: Command timeout in seconds
            
        Returns:
            Tuple of (output, exit_code)
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor, 
            self.execute_command, 
            pod_name, 
            command, 
            timeout
        )
    
    def get_pod_info(self, pod_name: str) -> Optional[Dict[str, Any]]:
        """
        Get information about a specific pod.
        
        Args:
            pod_name: Name of the pod
            
        Returns:
            Pod information dictionary or None if not found
        """
        return self.active_pods.get(pod_name)
    
    def list_active_pods(self) -> Dict[str, Dict[str, Any]]:
        """
        List all active pods managed by this manager.
        
        Returns:
            Dictionary of {pod_name: pod_info} for all active pods
        """
        return self.active_pods.copy()
    
    def cleanup_all_pods(self) -> int:
        """
        Kill all active pods managed by this manager.
        
        Returns:
            Number of pods successfully killed
        """
        killed_count = 0
        pod_names = list(self.active_pods.keys())
        
        for pod_name in pod_names:
            if self.kill_pod(pod_name):
                killed_count += 1
        
        print(f"[PodManager] Cleanup completed: {killed_count}/{len(pod_names)} pods killed")
        return killed_count
    
    def initialize_r2e_pod(self, pod_name: str):
        """
        Initialize R2E pod environment with required setup commands.
        
        Args:
            pod_name: Name of the pod to initialize
        """
        try:
            repo_path = "/testbed"
            alt_path = "/root"
            SKIP_FILES_NEW = [
                "run_tests.sh",
                "r2e_tests",
            ]
            
            # Create symbolic links for Python environment
            self.execute_command(pod_name, f"ln -s /testbed/.venv /root/.venv")
            self.execute_command(pod_name, f"ln -s {repo_path}/.venv/bin/python {alt_path}/.local/bin/python")
            self.execute_command(pod_name, f"ln -s {repo_path}/.venv/bin/python {alt_path}/.local/bin/python3")
            self.execute_command(pod_name, f"find {repo_path}/.venv/bin -type f -executable -exec ln -sf {{}} {alt_path}/.local/bin/ \\;")
            
            # Install chardet
            self.execute_command(pod_name, "uv pip install chardet")
            
            # Clean up Python cache files
            self.execute_command(pod_name, "find . -name '*.pyc' -delete")
            self.execute_command(pod_name, "find . -name '__pycache__' -exec rm -rf {} +")
            
            # Clean up cache from /r2e_tests
            self.execute_command(pod_name, "find /r2e_tests -name '*.pyc' -delete")
            self.execute_command(pod_name, "find /r2e_tests -name '__pycache__' -exec rm -rf {} +")
            
            # Move skip files to /root
            for skip_file in SKIP_FILES_NEW:
                self.execute_command(pod_name, f"mv {repo_path}/{skip_file} {alt_path}/{skip_file}")
            
            # Move r2e_tests to /root and create symbolic link
            self.execute_command(pod_name, f"mv /r2e_tests {alt_path}/r2e_tests")
            self.execute_command(pod_name, f"ln -s {alt_path}/r2e_tests {repo_path}/r2e_tests")
            
            print(f"[PodManager] R2E pod {pod_name} initialization completed successfully")
            
            for command_path in R2EGYM_COMMAND_FILES:
                #组合copy地址，去掉文件名称最后的.py部分
                dest_dir = "/usr/local/bin/" + os.path.basename(command_path).rsplit(".", 1)[0]
                self.kodo_runner.copy_to_container(pod_name, command_path, dest_dir)
                self.execute_command(pod_name, f"chmod +x {dest_dir}")
            
        except Exception as e:
            print(f"[PodManager] Error initializing R2E pod {pod_name}: {e}")
            # Don't raise the exception as initialization failure shouldn't stop trajectory execution
    
    def initialize_swebench_pod(self, pod_name: str):
        """
        Initialize SWE-bench pod environment with required setup commands.
        
        Args:
            pod_name: Name of the pod to initialize
        """
        try:
            # Make the run_tests.sh executable
            self.execute_command(pod_name, "chmod +x /run_tests.sh")
            
            # Make symlink of conda env to /root/.venv
            self.execute_command(pod_name, "ln -s /opt/miniconda3/envs/testbed /root/.venv")
            
            # Install required packages
            self.execute_command(pod_name, "python -m pip install chardet")
            
            print(f"[PodManager] SWE-bench pod {pod_name} initialization completed successfully")
            
        except Exception as e:
            print(f"[PodManager] Error initializing SWE-bench pod {pod_name}: {e}")
            # Don't raise the exception as initialization failure shouldn't stop trajectory execution
    
    def __del__(self):
        """Cleanup when the manager is destroyed."""
        try:
            self.cleanup_all_pods()
            if hasattr(self, 'executor'):
                self.executor.shutdown(wait=False, cancel_futures=True)
        except Exception as e:
            print(f"[PodManager] Error during cleanup: {e}")
