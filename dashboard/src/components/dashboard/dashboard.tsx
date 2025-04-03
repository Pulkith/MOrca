import React, { useState, useEffect } from 'react';
import { Activity, Power, RefreshCw, Trash2, Upload, AlertTriangle, CheckCircle, Clock, Server, Cpu, Maximize2, ChevronDown, ChevronUp, HardDrive, Zap, Thermometer, Database } from 'lucide-react';

const GPUClusterManagement = () => {
  // State for storing cluster data
  const [clusters, setClusters] = useState([]);
  const [expandedClusters, setExpandedClusters] = useState({});
  const [expandedGPUs, setExpandedGPUs] = useState({});
  const [selectedCluster, setSelectedCluster] = useState(null);
  const [isLoading, setIsLoading] = useState(true);
  const [systemMetrics, setSystemMetrics] = useState({
    totalGPUs: 0,
    activeJobs: 0,
    queuedJobs: 0,
    systemLoad: 0,
    memoryUtilization: 0,
  });

  // Mock data generation
  useEffect(() => {
    // Simulate API fetch delay
    setTimeout(() => {
      const mockClusters = generateMockClusters();
      setClusters(mockClusters);
      setSelectedCluster(mockClusters[0].id);
      
      // Calculate system metrics
      const metrics = calculateSystemMetrics(mockClusters);
      setSystemMetrics(metrics);
      
      setIsLoading(false);
    }, 1500);
  }, []);

  const generateMockClusters = () => {
    const mockClusters = [];
    const clusterTypes = ['DGX-A100', 'DGX-H100', 'RTX-8000'];
    
    for (let i = 0; i < 3; i++) {
      const clusterType = clusterTypes[i % clusterTypes.length];
      const gpuCount = clusterType === 'DGX-A100' ? 8 : clusterType === 'DGX-H100' ? 8 : 4;
      
      const gpus = [];
      for (let j = 0; j < gpuCount; j++) {
        const coreCount = clusterType === 'DGX-H100' ? 128 : clusterType === 'DGX-A100' ? 108 : 72;
        const memorySize = clusterType === 'DGX-H100' ? 80 : clusterType === 'DGX-A100' ? 40 : 48;
        
        const cores = [];
        for (let k = 0; k < coreCount; k += coreCount / 8) { // Group cores for display
          const runningTasks = Math.floor(Math.random() * 4);
          const queuedTasks = Math.floor(Math.random() * 6);
          
          cores.push({
            id: `core-${j}-${k}`,
            index: k,
            runningTasks,
            queuedTasks,
            utilization: Math.floor(Math.random() * 100),
            temperature: 40 + Math.floor(Math.random() * 40),
            coreGroup: `${k}-${k + (coreCount / 8) - 1}`,
          });
        }
        
        gpus.push({
          id: `gpu-${i}-${j}`,
          index: j,
          model: clusterType.replace('DGX-', ''),
          status: Math.random() > 0.1 ? 'online' : Math.random() > 0.5 ? 'warning' : 'error',
          utilization: Math.floor(Math.random() * 100),
          temperature: 50 + Math.floor(Math.random() * 40),
          memory: {
            total: memorySize,
            used: Math.floor(Math.random() * memorySize),
          },
          power: {
            current: 100 + Math.floor(Math.random() * 200),
            limit: 300 + Math.floor(Math.random() * 100),
          },
          cores,
          pcieBandwidth: Math.floor(Math.random() * 16) + 16,
          computeCapability: clusterType === 'DGX-H100' ? '9.0' : clusterType === 'DGX-A100' ? '8.0' : '7.5',
        });
      }
      
      mockClusters.push({
        id: `cluster-${i}`,
        name: `${clusterType}-Cluster-${i + 1}`,
        location: ['US-East', 'US-West', 'EU-Central'][i % 3],
        status: Math.random() > 0.8 ? 'warning' : 'online',
        gpus,
        totalMemory: gpuCount * (clusterType === 'DGX-H100' ? 80 : clusterType === 'DGX-A100' ? 40 : 48),
        uptime: Math.floor(Math.random() * 500) + 100,
        powerDraw: Math.floor(Math.random() * 5000) + 2000,
        networkedStorage: Math.floor(Math.random() * 500) + 500,
      });
    }
    
    return mockClusters;
  };

  const calculateSystemMetrics = (clusters) => {
    let totalGPUs = 0;
    let activeJobs = 0;
    let queuedJobs = 0;
    let totalUtilization = 0;
    let totalMemoryUsage = 0;
    let totalMemory = 0;
    
    clusters.forEach(cluster => {
      totalGPUs += cluster.gpus.length;
      
      cluster.gpus.forEach(gpu => {
        totalUtilization += gpu.utilization;
        totalMemoryUsage += gpu.memory.used;
        totalMemory += gpu.memory.total;
        
        gpu.cores.forEach(core => {
          activeJobs += core.runningTasks;
          queuedJobs += core.queuedTasks;
        });
      });
    });
    
    return {
      totalGPUs,
      activeJobs,
      queuedJobs,
      systemLoad: totalUtilization / totalGPUs,
      memoryUtilization: (totalMemoryUsage / totalMemory) * 100,
    };
  };

  const toggleClusterExpansion = (clusterId) => {
    setExpandedClusters(prev => ({
      ...prev,
      [clusterId]: !prev[clusterId]
    }));
  };

  const toggleGPUExpansion = (gpuId) => {
    setExpandedGPUs(prev => ({
      ...prev,
      [gpuId]: !prev[gpuId]
    }));
  };

  const getStatusColor = (status) => {
    switch (status) {
      case 'online':
        return 'text-green-400';
      case 'warning':
        return 'text-amber-400';
      case 'error':
        return 'text-red-500';
      default:
        return 'text-gray-400';
    }
  };

  const getStatusIcon = (status) => {
    switch (status) {
      case 'online':
        return <CheckCircle className="h-4 w-4 text-green-400" />;
      case 'warning':
        return <AlertTriangle className="h-4 w-4 text-amber-400" />;
      case 'error':
        return <AlertTriangle className="h-4 w-4 text-red-500" />;
      default:
        return <Clock className="h-4 w-4 text-gray-400" />;
    }
  };

  const getUtilizationColor = (value) => {
    if (value < 30) return 'bg-green-500';
    if (value < 70) return 'bg-amber-500';
    return 'bg-red-500';
  };

  const formatUptime = (hours) => {
    const days = Math.floor(hours / 24);
    const remainingHours = hours % 24;
    return `${days}d ${remainingHours}h`;
  };

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-screen bg-gray-900 text-gray-100">
        <div className="flex flex-col items-center">
          <Activity className="h-10 w-10 text-blue-500 animate-pulse" />
          <p className="mt-4 text-lg">Initializing Cluster Management System...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gray-900 text-gray-100 p-6">
      <header className="mb-6">
        <div className="flex justify-between items-center">
          <div className="flex items-center space-x-3">
            <Server className="h-8 w-8 text-blue-500" />
            <h1 className="text-2xl font-bold">NVIDIA GPU Cluster Management System</h1>
          </div>
          <div className="flex space-x-2">
            <button className="bg-gray-800 p-2 rounded hover:bg-gray-700 transition">
              <RefreshCw className="h-5 w-5 text-blue-400" />
            </button>
            <button className="bg-blue-600 p-2 rounded hover:bg-blue-500 transition">
              <Power className="h-5 w-5" />
            </button>
          </div>
        </div>
      </header>

      <div className="grid grid-cols-1 lg:grid-cols-4 gap-6 mb-6">
        <div className="bg-gray-800 p-4 rounded flex items-center">
          <div className="bg-blue-500/20 p-3 rounded mr-4">
            <Cpu className="h-6 w-6 text-blue-400" />
          </div>
          <div>
            <p className="text-gray-400 text-sm">Total GPUs</p>
            <p className="text-2xl font-bold">{systemMetrics.totalGPUs}</p>
          </div>
        </div>

        <div className="bg-gray-800 p-4 rounded flex items-center">
          <div className="bg-green-500/20 p-3 rounded mr-4">
            <Activity className="h-6 w-6 text-green-400" />
          </div>
          <div>
            <p className="text-gray-400 text-sm">Active Jobs</p>
            <p className="text-2xl font-bold">{systemMetrics.activeJobs}</p>
          </div>
        </div>

        <div className="bg-gray-800 p-4 rounded flex items-center">
          <div className="bg-amber-500/20 p-3 rounded mr-4">
            <Clock className="h-6 w-6 text-amber-400" />
          </div>
          <div>
            <p className="text-gray-400 text-sm">Queued Jobs</p>
            <p className="text-2xl font-bold">{systemMetrics.queuedJobs}</p>
          </div>
        </div>

        <div className="bg-gray-800 p-4 rounded flex items-center">
          <div className="bg-purple-500/20 p-3 rounded mr-4">
            <Zap className="h-6 w-6 text-purple-400" />
          </div>
          <div>
            <p className="text-gray-400 text-sm">System Load</p>
            <p className="text-2xl font-bold">{systemMetrics.systemLoad.toFixed(1)}%</p>
          </div>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        <div className="lg:col-span-3 bg-gray-800 rounded p-4">
          <h2 className="text-lg font-bold mb-4 flex items-center">
            <Server className="h-5 w-5 mr-2 text-blue-400" />
            Clusters
          </h2>
          <div className="space-y-2">
            {clusters.map(cluster => (
              <div 
                key={cluster.id} 
                className={`p-3 rounded cursor-pointer transition ${selectedCluster === cluster.id ? 'bg-blue-600' : 'bg-gray-700 hover:bg-gray-600'}`}
                onClick={() => setSelectedCluster(cluster.id)}
              >
                <div className="flex justify-between items-center">
                  <div className="flex items-center">
                    {getStatusIcon(cluster.status)}
                    <span className="ml-2 font-medium">{cluster.name}</span>
                  </div>
                  <span className="text-xs bg-gray-900 px-2 py-1 rounded">{cluster.gpus.length} GPUs</span>
                </div>
                <div className="mt-2 text-xs text-gray-400 flex justify-between">
                  <span>{cluster.location}</span>
                  <span>{formatUptime(cluster.uptime)} uptime</span>
                </div>
              </div>
            ))}
          </div>
        </div>

        <div className="lg:col-span-9 bg-gray-800 rounded p-4">
          {selectedCluster && (
            <>
              <div className="flex justify-between items-center mb-6">
                <div>
                  <h2 className="text-xl font-bold">{clusters.find(c => c.id === selectedCluster)?.name}</h2>
                  <div className="flex items-center mt-1 text-sm text-gray-400">
                    <span className="flex items-center mr-4">
                      <HardDrive className="h-4 w-4 mr-1" />
                      {clusters.find(c => c.id === selectedCluster)?.totalMemory} GB Total Memory
                    </span>
                    <span className="flex items-center mr-4">
                      <Zap className="h-4 w-4 mr-1" />
                      {clusters.find(c => c.id === selectedCluster)?.powerDraw} W Power
                    </span>
                    <span className="flex items-center">
                      <Database className="h-4 w-4 mr-1" />
                      {clusters.find(c => c.id === selectedCluster)?.networkedStorage} TB Storage
                    </span>
                  </div>
                </div>
                <div className="flex space-x-2">
                  <button className="bg-gray-700 hover:bg-gray-600 p-2 rounded transition flex items-center text-sm">
                    <RefreshCw className="h-4 w-4 mr-2" />
                    <span>Refresh</span>
                  </button>
                  <button className="bg-amber-600 hover:bg-amber-500 p-2 rounded transition flex items-center text-sm">
                    <Upload className="h-4 w-4 mr-2" />
                    <span>Offload</span>
                  </button>
                  <button className="bg-blue-600 hover:bg-blue-500 p-2 rounded transition flex items-center text-sm">
                    <Trash2 className="h-4 w-4 mr-2" />
                    <span>Clear</span>
                  </button>
                  <button className="bg-red-600 hover:bg-red-500 p-2 rounded transition flex items-center text-sm">
                    <Power className="h-4 w-4 mr-2" />
                    <span>Shutdown</span>
                  </button>
                </div>
              </div>

              <div className="space-y-4">
                {clusters.find(c => c.id === selectedCluster)?.gpus.map(gpu => (
                  <div key={gpu.id} className="bg-gray-700 rounded overflow-hidden">
                    <div 
                      className="p-3 flex items-center justify-between cursor-pointer hover:bg-gray-600 transition"
                      onClick={() => toggleGPUExpansion(gpu.id)}
                    >
                      <div className="flex items-center">
                        <div className={`h-3 w-3 rounded-full mr-3 ${getStatusColor(gpu.status)} bg-opacity-50`}></div>
                        <Cpu className="h-5 w-5 mr-2 text-blue-400" />
                        <span className="font-medium">GPU {gpu.index}: {gpu.model}</span>
                        <span className="ml-3 text-xs px-2 py-1 bg-gray-800 rounded">
                          CC {gpu.computeCapability}
                        </span>
                      </div>
                      <div className="flex items-center">
                        <div className="flex items-center mr-6">
                          <Thermometer className="h-4 w-4 mr-1 text-red-400" />
                          <span className={temperature => {
                            return temperature > 80 ? 'text-red-400' : temperature > 70 ? 'text-amber-400' : 'text-gray-300';
                          }}>{gpu.temperature}°C</span>
                        </div>
                        <div className="flex items-center mr-6">
                          <Zap className="h-4 w-4 mr-1 text-amber-400" />
                          <span>{gpu.power.current}W / {gpu.power.limit}W</span>
                        </div>
                        <div className="w-32 mr-4">
                          <div className="text-xs text-gray-400 mb-1 flex justify-between">
                            <span>Utilization</span>
                            <span>{gpu.utilization}%</span>
                          </div>
                          <div className="h-2 w-full bg-gray-800 rounded overflow-hidden">
                            <div 
                              className={`h-full ${getUtilizationColor(gpu.utilization)}`} 
                              style={{ width: `${gpu.utilization}%` }}
                            ></div>
                          </div>
                        </div>
                        <div className="w-32 mr-4">
                          <div className="text-xs text-gray-400 mb-1 flex justify-between">
                            <span>Memory</span>
                            <span>{gpu.memory.used}GB / {gpu.memory.total}GB</span>
                          </div>
                          <div className="h-2 w-full bg-gray-800 rounded overflow-hidden">
                            <div 
                              className={`h-full ${getUtilizationColor((gpu.memory.used / gpu.memory.total) * 100)}`} 
                              style={{ width: `${(gpu.memory.used / gpu.memory.total) * 100}%` }}
                            ></div>
                          </div>
                        </div>
                        <div className="flex items-center">
                          <Activity className="h-4 w-4 mr-1 text-purple-400" />
                          <span>{gpu.pcieBandwidth} GB/s</span>
                        </div>
                        <div className="ml-4">
                          {expandedGPUs[gpu.id] ? 
                            <ChevronUp className="h-5 w-5" /> : 
                            <ChevronDown className="h-5 w-5" />
                          }
                        </div>
                      </div>
                    </div>
                    
                    {expandedGPUs[gpu.id] && (
                      <div className="p-4 bg-gray-800 border-t border-gray-600">
                        <h4 className="text-sm font-medium mb-3 flex items-center">
                          <Maximize2 className="h-4 w-4 mr-2 text-blue-400" />
                          CUDA Core Groups
                        </h4>
                        <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
                          {gpu.cores.map(core => (
                            <div key={core.id} className="bg-gray-700 p-3 rounded">
                              <div className="flex justify-between items-center mb-2">
                                <span className="text-sm font-medium">Cores {core.coreGroup}</span>
                                <span className="text-xs bg-gray-800 px-2 py-1 rounded">{core.utilization}%</span>
                              </div>
                              <div className="flex justify-between text-xs text-gray-400 mb-2">
                                <span>Running Tasks</span>
                                <span className="font-medium text-green-400">{core.runningTasks}</span>
                              </div>
                              <div className="flex justify-between text-xs text-gray-400 mb-2">
                                <span>Queued Tasks</span>
                                <span className="font-medium text-amber-400">{core.queuedTasks}</span>
                              </div>
                              <div className="h-1.5 w-full bg-gray-800 rounded overflow-hidden">
                                <div 
                                  className={`h-full ${getUtilizationColor(core.utilization)}`} 
                                  style={{ width: `${core.utilization}%` }}
                                ></div>
                              </div>
                              <div className="mt-3 flex justify-end space-x-2">
                                <button className="p-1 bg-gray-600 rounded hover:bg-gray-500 transition">
                                  <Trash2 className="h-3 w-3 text-gray-300" />
                                </button>
                                <button className="p-1 bg-gray-600 rounded hover:bg-gray-500 transition">
                                  <RefreshCw className="h-3 w-3 text-gray-300" />
                                </button>
                              </div>
                            </div>
                          ))}
                        </div>
                      </div>
                    )}
                  </div>
                ))}
              </div>
            </>
          )}
        </div>
      </div>
    </div>
  );
};

export default GPUClusterManagement;