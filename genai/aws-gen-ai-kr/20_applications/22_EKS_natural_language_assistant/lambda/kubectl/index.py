import json
import logging
import os
import subprocess
import time

logger = logging.getLogger()
logger.setLevel(logging.INFO)

# these are coming from the kubectl layer
os.environ['PATH'] = '/opt/kubectl:/opt/awscli:' + os.environ['PATH']

outdir = os.environ.get('TEST_OUTDIR', '/tmp')
kubeconfig = os.path.join(outdir, 'kubeconfig')


def handler(event, context):
    logger.info(json.dumps(dict(event)))

    cluster_name = event['ClusterName']
    command = event['Command']

    # "log in" to the cluster
    subprocess.check_call(['aws', 'eks', 'update-kubeconfig',
                           '--name', cluster_name,
                           '--kubeconfig', kubeconfig
                           ])

    if os.path.isfile(kubeconfig):
        os.chmod(kubeconfig, 0o600)

    timeout_seconds = 10
    output = wait_for_output(command.split()[1:], int(timeout_seconds))
    logger.info(f"Response: {output}")

    return output


def wait_for_output(args, timeout_seconds):
    end_time = time.time() + timeout_seconds
    error = None

    while time.time() < end_time:
        try:
            # the output is surrounded with '', so we unquote
            output = kubectl(args).decode('utf-8')
            if output:
                return output
        except Exception as e:
            error = str(e)
            # also a recoverable error
            if 'NotFound' in error:
                pass
        time.sleep(10)

    raise RuntimeError(f'Timeout waiting for output from kubectl command: {args} (last_error={error})')


def kubectl(args):
    retry = 3
    while retry > 0:
        try:
            cmd = ['kubectl', '--kubeconfig', kubeconfig] + args
            output = subprocess.check_output(cmd, stderr=subprocess.PIPE)
        except subprocess.CalledProcessError as exc:
            output = exc.output + exc.stderr
            if b'i/o timeout' in output and retry > 0:
                logger.info("kubectl timed out, retries left: %s" % retry)
                retry = retry - 1
            else:
                raise Exception(output)
        else:
            logger.info(output)
            return output
