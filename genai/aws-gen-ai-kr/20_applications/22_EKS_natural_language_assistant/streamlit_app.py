import streamlit as st
import asyncio
import threading
import atexit
from InlineAgent.agent import InlineAgent
from InlineAgent.tools import MCPHttp
import pandas as pd
import altair as alt
import json
from datetime import datetime, UTC
import re
import nest_asyncio
from dotenv import load_dotenv
import os
import boto3


load_dotenv()
nest_asyncio.apply()

# Configuration
EC2_HOST = os.getenv('BASTION_HOST')
MCP_PORT = os.getenv('MCP_PORT')
AWS_REGION = os.getenv('AWS_REGION')
EKS_CLUSTER = os.getenv('EKS_CLUSTER')
LAMBDA_ARN = os.getenv('LAMBDA_ARN')

lambda_client = boto3.client("lambda")

# Page configuration
st.set_page_config(
    page_title="Kubernetes Cluster Assistant",
    page_icon="☸️",
    layout="wide"
)

# Initialize session state variables
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'agent' not in st.session_state:
    st.session_state.agent = None
if 'kubernetes_resources' not in st.session_state:
    st.session_state.kubernetes_resources = {
        'namespaces': [],
        'pods': [],
        'nodes': [],
        'deployments': [],
        'services': []
    }
if 'resource_timestamp' not in st.session_state:
    st.session_state.resource_timestamp = datetime.now()
if 'agent_lock' not in st.session_state:
    st.session_state.agent_lock = threading.Lock()
if 'is_agent_initialized' not in st.session_state:
    st.session_state.is_agent_initialized = False
if 'cluster_name' not in st.session_state:
    st.session_state.cluster_name = None
if 'mcp_client' not in st.session_state:
    st.session_state.mcp_client = None
if 'sse_task' not in st.session_state:
    st.session_state.sse_task = None
if 'event_loop' not in st.session_state:
    st.session_state.event_loop = asyncio.new_event_loop()
    asyncio.set_event_loop(st.session_state.event_loop)
if 'resource_stack' not in st.session_state:
    st.session_state.resource_stack = None


# Helper function to run async code properly in Streamlit
def run_async(coro):
    """Properly run coroutines within Streamlit"""
    loop = asyncio.get_event_loop()
    if loop.is_running():
        # If we're in a running event loop, use create_task
        return asyncio.create_task(coro)
    else:
        # Otherwise run in the session's event loop
        return st.session_state.event_loop.run_until_complete(coro)


async def initialize_agent():
    """Initialize the MCP client and InlineAgent"""
    try:
        from contextlib import AsyncExitStack

        # Create an exit stack to manage resources
        stack = AsyncExitStack()

        # Configure connection to the Kubernetes MCP server
        if st.session_state.mcp_client is None:
            client = await MCPHttp.create(
                url=f"http://{EC2_HOST}:{MCP_PORT}/sse",
                headers={},
                timeout=10,
                sse_read_timeout=300
            )
            # Register for proper cleanup
            st.session_state.mcp_client = client
            st.session_state.resource_stack = stack

        # Create the InlineAgent with the MCP client
        agent = InlineAgent(
            foundation_model="us.anthropic.claude-3-5-haiku-20241022-v1:0",
            instruction="""You are a Kubernetes cluster management assistant that helps users manage their EKS cluster.
            You have access to various kubectl commands through an MCP server.
            When users ask you about Kubernetes resources or want to perform actions, use the appropriate tools.
            Always show the relevant information clearly and explain what you're doing.
            """,
            agent_name="kubernetes-assistant",
            action_groups=[
                {
                    "name": "KubernetesActions",
                    "description": "Tools for managing Kubernetes clusters",
                    "mcp_clients": [st.session_state.mcp_client]
                }
            ]
        )

        # Add welcome message
        welcome_message = await agent.invoke(
            "Hello! I'm your Kubernetes assistant. How can I help you with your EKS cluster today?"
        )
        st.session_state.messages.append({"role": "assistant", "content": welcome_message})

        return agent
    except Exception as e:
        st.error(f"Error initializing agent: {str(e)}")
        return None


# Properly clean up resources
async def cleanup():
    """Clean up resources properly"""
    if hasattr(st.session_state, 'resource_stack'):
        await st.session_state.resource_stack.aclose()
    elif st.session_state.mcp_client:
        try:
            if hasattr(st.session_state.mcp_client, 'aclose'):
                await st.session_state.mcp_client.aclose()
        except Exception as e:
            print(f"Error closing MCP client: {str(e)}")


# Register cleanup function with proper event loop handling
atexit.register(lambda: run_async(cleanup()))


def get_eks_clusters():
    """Get list of EKS clusters using boto3"""
    return [EKS_CLUSTER]


def run_kubectl_command(command):
    """Run kubectl command and return the output"""
    event = {
        "ClusterName": EKS_CLUSTER,
        "Command": command
    }

    payload = json.dumps(event)

    try:
        response = lambda_client.invoke(
            FunctionName=LAMBDA_ARN,
            InvocationType='RequestResponse',
            Payload=payload
        )
        payload_bytes = response['Payload'].read()
        payload_str = payload_bytes.decode('utf-8')
        return json.loads(payload_str)

    except Exception as e:
        st.error(f"Error running kubectl command: {str(e)}")
        raise


@st.fragment
def fetch_kubernetes_resources():
    """Fetch Kubernetes resources using kubectl commands"""
    # Make sure we have a selected cluster
    if not st.session_state.cluster_name:
        clusters = get_eks_clusters()
        if clusters:
            st.session_state.cluster_name = clusters[0]
        else:
            st.error("No EKS clusters found")
            return

    try:
        # Fetch namespaces first
        namespaces_cmd = "kubectl get namespaces -o json"
        namespaces_output = run_kubectl_command(namespaces_cmd)
        namespaces_data = json.loads(namespaces_output) if namespaces_output else {"items": []}

        # Process namespaces data
        namespaces_list = []
        for namespace in namespaces_data.get("items", []):
            namespace_name = namespace.get("metadata", {}).get("name", "")
            if namespace_name:
                namespaces_list.append(namespace_name)

        # Add namespaces to kubernetes_resources dictionary
        # (We'll update the full dictionary at the end)

        # Fetch pods
        pods_cmd = "kubectl get pods --all-namespaces -o json"
        pods_output = run_kubectl_command(pods_cmd)
        pods_data = json.loads(pods_output) if pods_output else {"items": []}

        # Process pods data
        pods_list = []
        for pod in pods_data.get("items", []):
            pod_info = {
                "name": pod.get("metadata", {}).get("name", ""),
                "namespace": pod.get("metadata", {}).get("namespace", ""),
                "status": pod.get("status", {}).get("phase", ""),
                "ready": check_pod_ready(pod),
                "restarts": get_pod_restarts(pod),
                "age": calculate_age(pod.get("metadata", {}).get("creationTimestamp", "")),
            }
            pods_list.append(pod_info)

        # Fetch nodes
        nodes_cmd = "kubectl get nodes -o json"
        nodes_output = run_kubectl_command(nodes_cmd)
        nodes_data = json.loads(nodes_output) if nodes_output else {"items": []}

        # Process nodes data
        nodes_list = []
        for node in nodes_data.get("items", []):
            node_status = "Ready"
            for condition in node.get("status", {}).get("conditions", []):
                if condition.get("type") == "Ready" and condition.get("status") != "True":
                    node_status = "NotReady"

            node_info = {
                "name": node.get("metadata", {}).get("name", ""),
                "status": node_status,
                "roles": get_node_roles(node),
                "age": calculate_age(node.get("metadata", {}).get("creationTimestamp", "")),
                "version": node.get("status", {}).get("nodeInfo", {}).get("kubeletVersion", ""),
                "internal_ip": get_node_internal_ip(node),
                "instance_type": node.get("metadata", {}).get("labels", {}).get("node.kubernetes.io/instance-type", ""),
                "cpu_capacity": node.get("status", {}).get("capacity", {}).get("cpu", ""),
                "memory_capacity": node.get("status", {}).get("capacity", {}).get("memory", ""),
            }
            nodes_list.append(node_info)

        # Fetch deployments
        deployments_cmd = "kubectl get deployments --all-namespaces -o json"
        deployments_output = run_kubectl_command(deployments_cmd)
        deployments_data = json.loads(deployments_output) if deployments_output else {"items": []}

        # Process deployments data
        deployments_list = []
        for deployment in deployments_data.get("items", []):
            deployment_info = {
                "name": deployment.get("metadata", {}).get("name", ""),
                "namespace": deployment.get("metadata", {}).get("namespace", ""),
                "desired_replicas": deployment.get("spec", {}).get("replicas", 0),
                "available_replicas": deployment.get("status", {}).get("availableReplicas", 0),
                "ready_replicas": deployment.get("status", {}).get("readyReplicas", 0),
                "age": calculate_age(deployment.get("metadata", {}).get("creationTimestamp", ""))
            }
            deployments_list.append(deployment_info)

        # Fetch services
        services_cmd = "kubectl get services --all-namespaces -o json"
        services_output = run_kubectl_command(services_cmd)
        services_data = json.loads(services_output) if services_output else {"items": []}

        # Process services data
        services_list = []
        for service in services_data.get("items", []):
            service_info = {
                "name": service.get("metadata", {}).get("name", ""),
                "namespace": service.get("metadata", {}).get("namespace", ""),
                "type": service.get("spec", {}).get("type", ""),
                "cluster_ip": service.get("spec", {}).get("clusterIP", ""),
                "external_ip": get_external_ip(service),
                "ports": format_ports(service.get("spec", {}).get("ports", [])),
                "age": calculate_age(service.get("metadata", {}).get("creationTimestamp", ""))
            }
            services_list.append(service_info)

        # Update session state
        st.session_state.kubernetes_resources = {
            'namespaces': namespaces_list,
            'pods': pods_list,
            'nodes': nodes_list,
            'deployments': deployments_list,
            'services': services_list
        }
        st.session_state.resource_timestamp = datetime.now()

    except Exception as e:
        st.error(f"Error fetching Kubernetes resources: {str(e)}")


def check_pod_ready(pod):
    """Check if pod is ready"""
    container_statuses = pod.get("status", {}).get("containerStatuses", [])
    if not container_statuses:
        return "0/0"

    ready_count = sum(1 for status in container_statuses if status.get("ready", False))
    return f"{ready_count}/{len(container_statuses)}"


def get_pod_restarts(pod):
    """Get total pod restarts"""
    container_statuses = pod.get("status", {}).get("containerStatuses", [])
    if not container_statuses:
        return 0

    return sum(status.get("restartCount", 0) for status in container_statuses)


def get_node_roles(node):
    """Extract node roles from labels"""
    labels = node.get("metadata", {}).get("labels", {})
    roles = []

    for label in labels:
        if label.startswith("node-role.kubernetes.io/"):
            roles.append(label.split("/")[1])

    return ", ".join(roles) if roles else "worker"


def get_node_internal_ip(node):
    """Get node internal IP"""
    addresses = node.get("status", {}).get("addresses", [])
    for address in addresses:
        if address.get("type") == "InternalIP":
            return address.get("address", "")
    return ""


def get_external_ip(service):
    """Get external IP address of service"""
    if service.get("spec", {}).get("type") == "LoadBalancer":
        ingress = service.get("status", {}).get("loadBalancer", {}).get("ingress", [])
        if ingress and len(ingress) > 0:
            return ingress[0].get("hostname", "") or ingress[0].get("ip", "")
    return "N/A"


def format_ports(ports):
    """Format service ports"""
    port_strings = []
    for port in ports:
        protocol = port.get("protocol", "TCP")
        port_str = f"{port.get('port', '')}"
        if "targetPort" in port:
            port_str += f":{port.get('targetPort', '')}"
        port_str += f"/{protocol}"
        port_strings.append(port_str)

    return ", ".join(port_strings)


def calculate_age(timestamp_str):
    """Calculate age from timestamp"""
    if not timestamp_str:
        return ""

    try:
        created_time = datetime.strptime(timestamp_str, "%Y-%m-%dT%H:%M:%SZ")
        # Use timezone-aware datetime
        now = datetime.now(UTC)
        diff = now - created_time.replace(tzinfo=UTC)

        days = diff.days
        hours, remainder = divmod(diff.seconds, 3600)
        minutes, _ = divmod(remainder, 60)

        if days > 0:
            return f"{days}d"
        elif hours > 0:
            return f"{hours}h"
        else:
            return f"{minutes}m"
    except Exception:
        return ""


def process_user_input(user_input):
    """Process user input synchronously by properly managing async code"""
    with st.session_state.agent_lock:
        if not st.session_state.agent:
            st.session_state.agent = run_async(initialize_agent())
            st.session_state.is_agent_initialized = True

        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": user_input})

        try:
            # Get response from agent
            with st.spinner("Awaiting agent response"):
                response = run_async(st.session_state.agent.invoke(user_input))

            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": response})
        except Exception as e:
            error_message = f"Error: {str(e)}"
            st.session_state.messages.append({"role": "assistant", "content": error_message})


def create_resource_monitor():
    """Create visualizations for Kubernetes resources"""
    st.subheader("Kubernetes Resource Monitor")

    # Cluster selection
    clusters = get_eks_clusters()
    if clusters:
        selected_cluster = st.selectbox(
            "Select EKS Cluster",
            clusters,
            index=clusters.index(st.session_state.cluster_name) if st.session_state.cluster_name in clusters else 0
        )

        if selected_cluster != st.session_state.cluster_name:
            st.session_state.cluster_name = selected_cluster
            fetch_kubernetes_resources()
    else:
        st.warning("No EKS clusters found in region. Please check your AWS credentials and region settings.")
        return

    # Add refresh button and timestamp
    col1, col2 = st.columns([1, 3])
    with col1:
        if st.button("🔄 Refresh Data"):
            fetch_kubernetes_resources()
    with col2:
        st.text(f"Last updated: {st.session_state.resource_timestamp.strftime('%Y-%m-%d %H:%M:%S')}")

    # Create tabs for different resource types
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Namespaces", "Pods", "Nodes", "Deployments", "Services"])

    # Namespaces tab
    with tab1:
        if st.session_state.kubernetes_resources['namespaces']:
            try:
                # Convert namespace list to DataFrame for display
                namespace_df = pd.DataFrame(st.session_state.kubernetes_resources['namespaces'], columns=['Name'])

                st.dataframe(namespace_df, use_container_width=True)

                # Show namespace count
                st.metric("Total Namespaces", len(namespace_df))

                # Create namespace visualization
                if len(namespace_df) > 0:
                    # Count resources per namespace
                    namespace_resource_counts = {}

                    # Count pods per namespace
                    if st.session_state.kubernetes_resources['pods']:
                        pod_df = pd.DataFrame(st.session_state.kubernetes_resources['pods'])
                        pod_counts = pod_df['namespace'].value_counts().to_dict()
                        for ns in namespace_df['Name']:
                            namespace_resource_counts[ns] = namespace_resource_counts.get(ns, {})
                            namespace_resource_counts[ns]['pods'] = pod_counts.get(ns, 0)

                    # Count deployments per namespace
                    if st.session_state.kubernetes_resources['deployments']:
                        deployment_df = pd.DataFrame(st.session_state.kubernetes_resources['deployments'])
                        deployment_counts = deployment_df['namespace'].value_counts().to_dict()
                        for ns in namespace_df['Name']:
                            namespace_resource_counts[ns] = namespace_resource_counts.get(ns, {})
                            namespace_resource_counts[ns]['deployments'] = deployment_counts.get(ns, 0)

                    # Count services per namespace
                    if st.session_state.kubernetes_resources['services']:
                        service_df = pd.DataFrame(st.session_state.kubernetes_resources['services'])
                        service_counts = service_df['namespace'].value_counts().to_dict()
                        for ns in namespace_df['Name']:
                            namespace_resource_counts[ns] = namespace_resource_counts.get(ns, {})
                            namespace_resource_counts[ns]['services'] = service_counts.get(ns, 0)

                    # Create a dataframe for visualization
                    resource_data = []
                    for ns, counts in namespace_resource_counts.items():
                        for resource_type, count in counts.items():
                            resource_data.append({
                                'namespace': ns,
                                'resource_type': resource_type,
                                'count': count
                            })

                    if resource_data:
                        resource_df = pd.DataFrame(resource_data)

                        # Create a grouped bar chart
                        chart = alt.Chart(resource_df).mark_bar().encode(
                            x=alt.X('namespace:N', title='Namespace'),
                            y=alt.Y('count:Q', title='Count'),
                            color=alt.Color('resource_type:N', scale=alt.Scale(
                                domain=['pods', 'deployments', 'services'],
                                range=['#4CAF50', '#2196F3', '#FF9800']
                            )),
                            tooltip=['namespace', 'resource_type', 'count']
                        ).properties(title='Resources by Namespace')

                        st.altair_chart(chart, use_container_width=True)

            except Exception as e:
                st.error(f"Error rendering namespace data: {str(e)}")
        else:
            st.info("No namespace data available. Click Refresh to fetch data.")

    # Pod metrics
    with tab2:
        if st.session_state.kubernetes_resources['pods']:
            try:
                pod_df = pd.DataFrame(st.session_state.kubernetes_resources['pods'])

                # Filter by namespace - use the namespaces from kubernetes_resources
                available_namespaces = ['All namespaces']
                if st.session_state.kubernetes_resources['namespaces']:
                    available_namespaces += sorted(st.session_state.kubernetes_resources['namespaces'])
                else:
                    # Fallback to unique namespaces from pods if namespace list is empty
                    available_namespaces += sorted(pod_df['namespace'].unique().tolist())

                selected_namespace = st.selectbox("Filter by namespace", available_namespaces)

                if selected_namespace != 'All namespaces':
                    filtered_df = pod_df[pod_df['namespace'] == selected_namespace]
                else:
                    filtered_df = pod_df

                # Search filter
                search_term = st.text_input("Search pods")
                if search_term:
                    filtered_df = filtered_df[filtered_df['name'].str.contains(search_term, case=False)]

                st.dataframe(filtered_df, use_container_width=True)

                # Create status chart
                status_count = filtered_df['status'].value_counts().reset_index()
                status_count.columns = ['status', 'count']

                if not status_count.empty:
                    status_chart = alt.Chart(status_count).mark_arc().encode(
                        theta=alt.Theta(field="count", type="quantitative"),
                        color=alt.Color(field="status", type="nominal", scale=alt.Scale(
                            domain=['Running', 'Pending', 'Failed', 'Succeeded', 'Unknown'],
                            range=['#4CAF50', '#FFC107', '#F44336', '#2196F3', '#9E9E9E']
                        )),
                        tooltip=['status', 'count']
                    ).properties(title='Pod Status Distribution', height=300)

                    st.altair_chart(status_chart, use_container_width=True)
            except Exception as e:
                st.error(f"Error rendering pod data: {str(e)}")
        else:
            st.info("No pod data available. Click Refresh to fetch data.")

    # Node metrics
    with tab3:
        if st.session_state.kubernetes_resources['nodes']:
            try:
                node_df = pd.DataFrame(st.session_state.kubernetes_resources['nodes'])
                st.dataframe(node_df, use_container_width=True)

                # Create node info visualization
                if len(node_df) > 0:
                    # Create a bar chart showing node capacity
                    node_df['cpu_numeric'] = pd.to_numeric(node_df['cpu_capacity'], errors='coerce')

                    # Clean memory capacity (remove Ki, Mi, etc.)
                    node_df['memory_numeric'] = node_df['memory_capacity'].apply(
                        lambda x: float(re.sub(r'[A-Za-z]+', '', x)) if isinstance(x, str) else x
                    )

                    col1, col2 = st.columns(2)
                    with col1:
                        cpu_chart = alt.Chart(node_df).mark_bar().encode(
                            x=alt.X('name:N', title='Node Name'),
                            y=alt.Y('cpu_numeric:Q', title='CPU Capacity (cores)'),
                            color=alt.Color('status:N', scale=alt.Scale(
                                domain=['Ready', 'NotReady'],
                                range=['#4CAF50', '#F44336']
                            ))
                        ).properties(title='Node CPU Capacity')
                        st.altair_chart(cpu_chart, use_container_width=True)

                    with col2:
                        memory_chart = alt.Chart(node_df).mark_bar().encode(
                            x=alt.X('name:N', title='Node Name'),
                            y=alt.Y('memory_numeric:Q', title='Memory Capacity'),
                            color=alt.Color('status:N', scale=alt.Scale(
                                domain=['Ready', 'NotReady'],
                                range=['#4CAF50', '#F44336']
                            ))
                        ).properties(title='Node Memory Capacity')
                        st.altair_chart(memory_chart, use_container_width=True)

                    # Show node details in expandable sections
                    for _, node in node_df.iterrows():
                        with st.expander(f"Node: {node['name']}"):
                            col1, col2 = st.columns(2)
                            with col1:
                                st.write("**Status:**", node['status'])
                                st.write("**Roles:**", node['roles'])
                                st.write("**Internal IP:**", node['internal_ip'])
                            with col2:
                                st.write("**Kubernetes Version:**", node['version'])
                                st.write("**Instance Type:**", node['instance_type'])
                                st.write("**Age:**", node['age'])
            except Exception as e:
                st.error(f"Error rendering node data: {str(e)}")
        else:
            st.info("No node data available. Click Refresh to fetch data.")

    # Deployment metrics
    with tab4:
        if st.session_state.kubernetes_resources['deployments']:
            try:
                deployment_df = pd.DataFrame(st.session_state.kubernetes_resources['deployments'])

                # Filter by namespace - use the namespaces from kubernetes_resources
                available_namespaces = ['All namespaces']
                if st.session_state.kubernetes_resources['namespaces']:
                    available_namespaces += sorted(st.session_state.kubernetes_resources['namespaces'])
                else:
                    # Fallback to unique namespaces from deployments if namespace list is empty
                    available_namespaces += sorted(deployment_df['namespace'].unique().tolist())

                selected_namespace = st.selectbox("Filter deployments by namespace", available_namespaces,
                                                  key="deployment_namespace")

                if selected_namespace != 'All namespaces':
                    filtered_df = deployment_df[deployment_df['namespace'] == selected_namespace]
                else:
                    filtered_df = deployment_df

                st.dataframe(filtered_df, use_container_width=True)

                # Create replicas chart
                if len(filtered_df) > 0:
                    # Melt the dataframe for easier plotting
                    plot_columns = ['desired_replicas', 'available_replicas', 'ready_replicas']
                    available_columns = [col for col in plot_columns if col in filtered_df.columns]

                    if available_columns:
                        melted_df = pd.melt(
                            filtered_df,
                            id_vars=['name'],
                            value_vars=available_columns,
                            var_name='replica_type',
                            value_name='count'
                        )

                        # Create a nicer legend
                        melted_df['replica_type'] = melted_df['replica_type'].map({
                            'desired_replicas': 'Desired',
                            'available_replicas': 'Available',
                            'ready_replicas': 'Ready'
                        })

                        replica_chart = alt.Chart(melted_df).mark_bar().encode(
                            x=alt.X('name:N', title='Deployment Name'),
                            y=alt.Y('count:Q', title='Replica Count'),
                            color=alt.Color('replica_type:N', scale=alt.Scale(
                                domain=['Desired', 'Available', 'Ready'],
                                range=['#2196F3', '#4CAF50', '#FFC107']
                            ))
                        ).properties(title='Deployment Replicas')

                        st.altair_chart(replica_chart, use_container_width=True)
            except Exception as e:
                st.error(f"Error rendering deployment data: {str(e)}")
        else:
            st.info("No deployment data available. Click Refresh to fetch data.")

    # Service metrics
    with tab5:
        if st.session_state.kubernetes_resources['services']:
            try:
                service_df = pd.DataFrame(st.session_state.kubernetes_resources['services'])

                # Filter by namespace - use the namespaces from kubernetes_resources
                available_namespaces = ['All namespaces']
                if st.session_state.kubernetes_resources['namespaces']:
                    available_namespaces += sorted(st.session_state.kubernetes_resources['namespaces'])
                else:
                    # Fallback to unique namespaces from services if namespace list is empty
                    available_namespaces += sorted(service_df['namespace'].unique().tolist())

                selected_namespace = st.selectbox("Filter services by namespace", available_namespaces,
                                                  key="service_namespace")

                if selected_namespace != 'All namespaces':
                    filtered_df = service_df[service_df['namespace'] == selected_namespace]
                else:
                    filtered_df = service_df

                st.dataframe(filtered_df, use_container_width=True)

                # Create a service type chart
                if 'type' in filtered_df.columns and len(filtered_df) > 0:
                    type_count = filtered_df['type'].value_counts().reset_index()
                    type_count.columns = ['type', 'count']

                    type_chart = alt.Chart(type_count).mark_arc().encode(
                        theta=alt.Theta(field="count", type="quantitative"),
                        color=alt.Color(field="type", type="nominal", scale=alt.Scale(
                            domain=['ClusterIP', 'NodePort', 'LoadBalancer', 'ExternalName'],
                            range=['#2196F3', '#4CAF50', '#9C27B0', '#FF9800']
                        )),
                        tooltip=['type', 'count']
                    ).properties(title='Service Types')

                    st.altair_chart(type_chart, use_container_width=True)
            except Exception as e:
                st.error(f"Error rendering service data: {str(e)}")
        else:
            st.info("No service data available. Click Refresh to fetch data.")


@st.fragment
def create_chat_interface():
    """Create the chat interface"""
    st.subheader("Kubernetes Assistant Chat")

    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    # Chat input
    if user_input := st.chat_input("Ask something about your Kubernetes cluster..."):
        # Process the input using our synchronized wrapper
        process_user_input(user_input)
        st.rerun()


def main():
    """Main function to run the Streamlit app"""
    st.title("☸️ Kubernetes Cluster Manager")

    # Initialize agent if not already done
    if not st.session_state.is_agent_initialized:
        with st.spinner("Initializing Kubernetes assistant..."):
            # Use run_async to properly manage the event loop
            st.session_state.agent = run_async(initialize_agent())
            st.session_state.is_agent_initialized = True

    # Initial resource fetch if needed
    if not st.session_state.kubernetes_resources['pods'] and not st.session_state.kubernetes_resources['nodes']:
        with st.spinner("Fetching Kubernetes resources..."):
            fetch_kubernetes_resources()

    # Create two columns for layout
    col1, col2 = st.columns([0.4, 0.6])

    # Chat interface in the left column
    with col1:
        create_chat_interface()

    # Resource monitor in the right column
    with col2:
        create_resource_monitor()


if __name__ == "__main__":
    main()
