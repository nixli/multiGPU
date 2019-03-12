#!/usr/bin/env python2.7
import os
import time
import sys
os.system("python -m pip install boto")
import boto.ec2



def deploy_aws():

    print "*"*200
    print "This script requires to set a environment variable called: ACCESS_FILE that contains login credentials of your AWS account, please make that file local and never share it to others"
    print "*"*200
    filename = os.environ['ACCESS_FILE']

    with open(filename) as f:
        mylist = f.read().splitlines()

    try:
        aws_access_key_id = mylist[0]
        aws_secret_access_key=mylist[1]
        region = 'us-east-1'
        #create the connection to the AWS platform
        connection = boto.ec2.connect_to_region(region,aws_access_key_id=aws_access_key_id,
                                                aws_secret_access_key =aws_secret_access_key)
        f.close()
    except Exception as er:
        print "there is an error in your keyfile, error: "
        print er
        exit(-1)

    if os.path.exists("./multiGPU.pem"):
        os.remove("./multiGPU.pem")      
    connection.delete_key_pair('multiGPU')
    ssh_key_pair =  connection.create_key_pair('multiGPU')
    #save this permission under the current derectory, not committed
    ssh_key_pair.save('./')
    os.chmod("./multiGPU.pem", int('400', base=8))
    print "\ncreated a permission file for your instance, saved in the current directory"

    try:
        connection.delete_security_group('multiGPU_deploy')
        security_group = connection.create_security_group(name ='multiGPU_deploy', description='used for setting up multiGPU instance on aws')
        #authorize some access of this group, so we could ssh and browes the instance later on
        security_group.authorize(ip_protocol='ICMP', from_port=-1, to_port=-1, cidr_ip='0.0.0.0/0')
        security_group.authorize(ip_protocol='TCP', from_port=22, to_port=22, cidr_ip='0.0.0.0/0')
        security_group.authorize(ip_protocol='TCP', from_port=80, to_port=80, cidr_ip='0.0.0.0/0')

    except Exception:
        x = connection.get_all_security_groups(groupnames=['multiGPU_deploy'])
        if not x:
            return
        security_group = x[0]

 
    #print the id for futher referenc
    print "created/retrived a security group for your instance with id: ", security_group.id

    print "creating instance..."
    #create the instance with the parameters we set up, and some addtional options
    aws_instance = connection.run_instances('ami-d3515da8', min_count=1, max_count=1,
                  key_name='multiGPU', security_groups=['multiGPU_deploy'],
                  addressing_type=None, instance_type='t2.nano', placement=None,
                  kernel_id=None, ramdisk_id=None, monitoring_enabled=True, subnet_id=None,
                  block_device_map=None, disable_api_termination=False, instance_initiated_shutdown_behavior=None,
                  private_ip_address=None, placement_group=None, client_token=None, security_group_ids=[security_group.id],
                  additional_info={"description" :"Created by Xin for Multiple GPU Envirnment with CUDA"}, instance_profile_name=None, instance_profile_arn=None, tenancy=None,
                  ebs_optimized=False, network_interfaces=None)

    #now we need to assign a static ip to the instance
    running_instance=aws_instance.instances[0]
    while running_instance.state_code!=16:
        print ". ",
        sys.stdout.flush()
        time.sleep(2)
        running_instance.update()

    print "\nDone!\nDNS Name: ", running_instance.public_dns_name

if __name__ == "__main__":
    deploy_aws()
