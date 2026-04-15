import multiprocessing as mp
from typing import Callable, Any
import cloudpickle

class CloudpickleWrapper:
    """Uses cloudpickle to serialize contents (useful for lambdas/env_fns)."""
    def __init__(self, x: Any):
        self.x = x

    def __getstate__(self) -> dict[str, Any]:
        return {"x": cloudpickle.dumps(self.x)}

    def __setstate__(self, ob: dict[str, Any]) -> None:
        self.x = cloudpickle.loads(ob["x"])
        
    def var(self):
        return self.x

def _worker_loop(remote: mp.connection.Connection, parent_remote: mp.connection.Connection, env_fn_wrapper: CloudpickleWrapper):
    """
    Worker process loop for stepping a single multi-agent environment (BangaloreCorridor).
    """
    parent_remote.close()
    env = env_fn_wrapper.var()()
    
    try:
        while True:
            cmd, data = remote.recv()
            if cmd == "step":
                # data is a dict: {tls_id: action}
                obs, reward, terminated, truncated, info = env.step(data)
                
                # Piggyback the real global state so the main process can use it
                # for centralized critics / manager networks without extra RPCs.
                if hasattr(env, "get_global_state"):
                    info["global_state"] = env.get_global_state()
                
                # Auto-reset logic: if terminated or truncated, we must return the new observation
                if terminated or truncated:
                    # SB3 VecEnv convention: store terminal obs inside "terminal_observation" 
                    # Note: Since obs here is a dict of all agents, we store the full dict
                    info["terminal_observation"] = obs
                    obs, reset_info = env.reset()
                    if hasattr(env, "get_global_state"):
                        info["new_global_state"] = env.get_global_state()
                remote.send((obs, reward, terminated, truncated, info))
                
            elif cmd == "reset":
                obs, info = env.reset()
                if hasattr(env, "get_global_state"):
                    info["global_state"] = env.get_global_state()
                remote.send((obs, info))
                
            elif cmd == "set_scale":
                env.scale = data
                remote.send(True)
                
            elif cmd == "close":
                env.close()
                remote.close()
                break
                
            elif cmd == "get_attr":
                remote.send(getattr(env, data))
                
            else:
                raise NotImplementedError(f"`{cmd}` is not implemented in the worker")
    except KeyboardInterrupt:
        print("Worker KeyboardInterrupt")
    except Exception as e:
        print(f"Worker Error: {e}")
        raise e
    finally:
        env.close()

class MultiAgentSharedSubproc:
    """
    A unified manager for 4 Subproc Python workers, each holding a multi-agent master environment.
    This routes dictionaries of (agent_id: action) to the workers via multiprocess Pipes.
    """
    def __init__(self, env_fns: list[Callable]):
        self.num_envs = len(env_fns)
        
        self.remotes, self.work_remotes = zip(*[mp.Pipe() for _ in range(self.num_envs)])
        self.processes = []
        
        for work_remote, remote, env_fn in zip(self.work_remotes, self.remotes, env_fns):
            process = mp.Process(
                target=_worker_loop,
                args=(work_remote, remote, CloudpickleWrapper(env_fn)),
                daemon=True,
            )
            process.start()
            self.processes.append(process)
            work_remote.close()
            
    def step_async(self, actions_list: list[dict]):
        """actions_list: [ {J1: a1_env0, J2: a2_env0}, {J1: a1_env1, J2: a2_env1} ... ]"""
        for remote, actions in zip(self.remotes, actions_list):
            remote.send(("step", actions))
            
    def step_wait(self) -> list[tuple]:
        """Returns: [ (obs_dict, reward_dict, terminated, truncated, info_dict), ... ]"""
        return [remote.recv() for remote in self.remotes]
        
    def reset(self) -> list[tuple]:
        """Returns: [ (obs_dict, info_dict), ... ]"""
        for remote in self.remotes:
            remote.send(("reset", None))
        return [remote.recv() for remote in self.remotes]
        
    def set_scale(self, scale: float):
        for remote in self.remotes:
            remote.send(("set_scale", scale))
        for remote in self.remotes:
            remote.recv()
            
    def close(self):
        for remote in self.remotes:
            remote.send(("close", None))
        for process in self.processes:
            process.join()
