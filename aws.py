#!/usr/bin/env python2.7
import os
import time
import sys
import boto.ec2
import argparse
# establish an ec2 instance in the region
def establish_connection_with_key( region, filename):

    print "*"*200
    print "This script requires to set a environment variable called: ACCESS_FILE that contains login credentials of your AWS account, please make that file local and never share it to others"
    print "*"*200
    # filename = os.environ['ACCESS_FILE']

    with open(filename) as f:
        mylist = f.read().splitlines()

    try:
        aws_access_key_id = mylist[0]
        aws_secret_access_key=mylist[1]
        #create the connection to the AWS platform
        connection = boto.ec2.connect_to_region(region,aws_access_key_id=aws_access_key_id,
                                                aws_secret_access_key =aws_secret_access_key)
        f.close()
    except Exception as er:
        print "there is an error in your keyfile, error: "
        print er
        exit(-1)

    return connection


# terminate the instance id
def terminate_aws(connection, instanceID):
    for instance in connection.get_only_instances():
        if instance.id==instanceID:

            if instance.state_code!=48: #not terminated
                try:
                    instance.terminate()
                    print "terminating ",
                    while instance.state_code!=48:
                        print ". ",
                        sys.stdout.flush()
                        time.sleep(2)
                        instance.update()
                    print("\ninstance {} successfully terminated".format(instanceID))
                    exit(0)
                except Exception as er:
                    print "termination unsuccessful: "
                    print er
                    exit(-1)
            else:
                print("instance already terminated")
                exit(0)

    print("No instance matches the id provided")
    exit(-2)

def deploy_aws(connection, it, ai):


    if os.path.exists("./multiGPU.pem"):
        os.remove("./multiGPU.pem")      
    connection.delete_key_pair('multiGPU')
    ssh_key_pair =  connection.create_key_pair('multiGPU')
    #save this permission under the current derectory, not committed
    ssh_key_pair.save('./')
    os.chmod("./multiGPU.pem", int('400', base=8))
    print "\nCreated a permission file for your instance, saved in the current directory"

    try:
        x = connection.get_all_security_groups(groupnames=['multiGPU_deploy'])
        security_group = x[0]
    except boto.exception.EC2ResponseError  as e:
          security_group = connection.create_security_group(name ='multiGPU_deploy', description='used for setting up multiGPU instance on aws')
          # authorize some access of this group, so we could ssh and browes the instance later on
          security_group.authorize(ip_protocol='ICMP', from_port=-1, to_port=-1, cidr_ip='0.0.0.0/0')
          security_group.authorize(ip_protocol='TCP', from_port=22, to_port=22, cidr_ip='0.0.0.0/0')
          security_group.authorize(ip_protocol='TCP', from_port=80, to_port=80, cidr_ip='0.0.0.0/0')

    print "Created/retrived a security group for your instance with id: ", security_group.id
    # create the instance with the parameters we set up, and some addtional options
    aws_instance = connection.run_instances(ai, min_count=1, max_count=1,
                  key_name='multiGPU', security_groups=['multiGPU_deploy'],
                  addressing_type=None, instance_type=it, placement=None,
                  kernel_id=None, ramdisk_id=None, monitoring_enabled=True, subnet_id=None,
                  block_device_map=None, disable_api_termination=False, instance_initiated_shutdown_behavior=None,
                  private_ip_address=None, placement_group=None, client_token=None, security_group_ids=[security_group.id],
                  additional_info={"description" :"Created by Xin for Multiple GPU Envirnment with CUDA"}, instance_profile_name=None, instance_profile_arn=None, tenancy=None,
                  ebs_optimized=False, network_interfaces=None)

    # now we need to assign a static ip to the instance
    running_instance=aws_instance.instances[0]
    print "\n Instance ID: ", running_instance.id
    print "Creating instance..."
    while running_instance.state_code!=16:
        print ". ",
        sys.stdout.flush()
        time.sleep(2)
        running_instance.update()

    print "\nDone!\nDNS Name: ", running_instance.public_dns_name
    return running_instance.id
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Script to launch aws instance and terminate easily')

    parser.add_argument('access_file', type=str,
                    help='file that stores the access ID and Key in plain text, in the form of\
                            \nxxxxxxxxx<access id>\nxxxxxxxxxx<access key>')
    parser.add_argument('--region', dest='region', action='store', default='us-east-2',
                    help='the region to connect to')

    parser.add_argument('--terminate', dest='terminate', action='store_true',
                    help='whether to terminate an instance given by --instance')

    parser.add_argument('--instance', dest='instance', default=None,
                    help='Instance ID to terminate if --terminate is set')

    parser.add_argument('--instance-type', dest='instance_type', type=str, default='t2.micro', 
                    help = 'The type of instance, defaul is t2.micro')

    parser.add_argument('--ami', dest='ami_id', type=str, default='ami-02bcbb802e03574ba', 
                    help = 'The ami id of the instnce, defaul is Amazon Linux 2 AMI (HVM), SSD Volume Type"ami-02bcbb802e03574ba"')

    args = parser.parse_args()

    if args.terminate and not args.instance:
        print "Please provide an instance to terminate"
        exit(-1)
    connection = establish_connection_with_key(args.region, args.access_file)
    if args.terminate:
        terminate_aws(connection, args.instance)
    else:
        deploy_aws(connection, args.instance_type, args.ami_id) 
