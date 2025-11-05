# This command sets the default compute region to us-west3 for all subsequent gcloud commands.
# This configures the gcloud CLI to save you from typing the --region flag repeatedly.
gcloud config set compute/region us-west3

# This command sets the default compute zone to us-west3-c for all subsequent gcloud commands.
# This configures the gcloud CLI to save you from typing the --zone flag repeatedly.
gcloud config set compute/zone us-west3-c

# This command creates a new virtual machine instance named web1 in the us-west3-c zone.
# The created object is a Compute Engine VM instance, which functions as a web server in this script.
gcloud compute instances create web1 \
  --zone=us-west3-c \
  --tags=network-lb-tag \
  --machine-type=e2-small \
  --image-family=debian-12 \
  --image-project=debian-cloud \
  --metadata=startup-script='#!/bin/bash
    apt-get update
    apt-get install apache2 -y
    service apache2 restart
    echo "
<h3>Web Server: web1</h3>" | tee /var/www/html/index.html'

# This command creates a new virtual machine instance named web2 in the us-west3-c zone.
# The created object is a Compute Engine VM instance, which functions as a web server in this script.
gcloud compute instances create web2 \
  --zone=us-west3-c \
  --tags=network-lb-tag \
  --machine-type=e2-small \
  --image-family=debian-12 \
  --image-project=debian-cloud \
  --metadata=startup-script='#!/bin/bash
    apt-get update
    apt-get install apache2 -y
    service apache2 restart
    echo "
<h3>Web Server: web2</h3>" | tee /var/www/html/index.html'

# This command creates a new virtual machine instance named web3 in the us-west3-c zone.
# The created object is a Compute Engine VM instance, which functions as a web server in this script.
gcloud compute instances create web3 \
  --zone=us-west3-c \
  --tags=network-lb-tag \
  --machine-type=e2-small \
  --image-family=debian-12 \
  --image-project=debian-cloud \
  --metadata=startup-script='#!/bin/bash
    apt-get update
    apt-get install apache2 -y
    service apache2 restart
    echo "
<h3>Web Server: web3</h3>" | tee /var/www/html/index.html'


# This command creates a firewall rule named www-firewall-network-lb to allow incoming TCP traffic on port 80.
# The created object is a Firewall Rule, which controls traffic to and from VM instances based on tags.
gcloud compute firewall-rules create www-firewall-network-lb \
    --target-tags network-lb-tag --allow tcp:80

# This command creates a regional external IP address named network-lb-ip-1 in the us-west3 region.
# The created object is a static External IP Address, providing a fixed public entry point for the load balancer.
gcloud compute addresses create network-lb-ip-1 \
  --region us-west3

# This command creates an HTTP health check named basic-check to monitor the health of instances.
# The created object is an HTTP Health Check, used by the load balancer to send requests to instances to see if they are responsive.
gcloud compute http-health-checks create basic-check

# This command creates a target pool named www-pool in the us-west3 region for the load balancer.
# The created object is a Target Pool, which holds a group of instances that will receive incoming traffic from the forwarding rule.
gcloud compute target-pools create www-pool \
  --region us-west3 --http-health-check basic-check


# This command adds the instances web1, web2, and web3 to the www-pool target pool.
# This modifies the Target Pool by adding instances to it, making them eligible to receive traffic.
gcloud compute target-pools add-instances www-pool \
    --instances web1,web2,web3

# This command removes the instances web1, web2, and web3 from the www-pool target pool.
# This modifies the Target Pool by removing instances, taking them out of the load balancing rotation.
gcloud compute target-pools remove-instances www-pool \
    --instances www1,www2,www3

# This command creates a forwarding rule named www-rule to forward traffic from the public IP to the target pool.
# The created object is a Forwarding Rule, which matches traffic by IP address and port and directs it to a target.
gcloud compute forwarding-rules create www-rule \
    --region  us-west3 \
    --ports 80 \
    --address network-lb-ip-1 \
    --target-pool www-pool


# ==============================
# This command creates an instance template named lb-backend-template for creating identical backend instances.
# The created object is an Instance Template, which is a blueprint for creating VMs, ensuring they are all configured the same way.
gcloud compute instance-templates create lb-backend-template \
   --region=us-west3 \
   --network=default \
   --subnet=default \
   --tags=allow-health-check \
   --machine-type=e2-small \
   --image-family=debian-12 \
   --image-project=debian-cloud \
   --metadata=startup-script='#!/bin/bash
     apt-get update
     apt-get install apache2 -y
     a2ensite default-ssl
     a2enmod ssl
     vm_hostname="$(curl -H "Metadata-Flavor:Google" \
     http://169.254.169.254/computeMetadata/v1/instance/name)"
     echo "Page served from: $vm_hostname" | \
     tee /var/www/html/index.html
     systemctl restart apache2'

# This command creates a managed instance group named lb-backend-group with 2 instances, based on the lb-backend-template.
# The created object is a Managed Instance Group (MIG), which automatically manages a fleet of identical VMs, handling scaling and auto-healing.
gcloud compute instance-groups managed create lb-backend-group \
   --template=lb-backend-template --size=2 --zone=us-west3-c

# This command creates a firewall rule to allow health check traffic from Google Cloud health checkers.
# The created object is a Firewall Rule, specifically designed to allow probes from Google's health checking infrastructure to reach your instances.
gcloud compute firewall-rules create fw-allow-health-check \
  --network=default \
  --action=allow \
  --direction=ingress \
  --source-ranges=130.211.0.0/22,35.191.0.0/16 \
  --target-tags=allow-health-check \
  --rules=tcp:80

# This command creates a global static external IPv4 address for the load balancer.
# The created object is a global External IP Address, providing a single anycast IP for users to access the service worldwide.
gcloud compute addresses create lb-ipv4-1 \
  --ip-version=IPV4 \
  --global

# This command retrieves and displays the created global IP address.
# This does not create an object, but rather queries and outputs the IP address of the 'lb-ipv4-1' address object.
gcloud compute addresses describe lb-ipv4-1 \
  --format="get(address)" \
  --global

# This command creates an HTTP health check to monitor the backend services.
# The created object is an HTTP Health Check, used by the Backend Service to determine the health of instances within the instance group.
gcloud compute health-checks create http http-basic-check \
  --port 80

# This command creates a backend service to manage the backend instances.
# The created object is a Backend Service, which defines how traffic is distributed to the backends and which health checks to use.
gcloud compute backend-services create web-backend-service \
  --protocol=HTTP \
  --port-name=http \
  --health-checks=http-basic-check \
  --global

# This command adds the managed instance group to the backend service.
# This modifies the Backend Service by linking it to the group of instances that should handle the traffic.
gcloud compute backend-services add-backend web-backend-service \
  --instance-group=lb-backend-group \
  --instance-group-zone=us-west3-c \
  --global

# This command creates a URL map to route incoming requests to the backend service.
# The created object is a URL Map, which defines the rules for routing requests to different backend services based on the request's URL.
gcloud compute url-maps create web-map-http \
    --default-service web-backend-service

# This command creates an HTTP proxy to receive requests and route them to the URL map.
# The created object is a Target HTTP Proxy, which terminates the HTTP session and uses the URL map to direct the request.
gcloud compute target-http-proxies create http-lb-proxy \
    --url-map web-map-http

# This command creates a global forwarding rule to route incoming requests to the HTTP proxy.
# The created object is a Global Forwarding Rule, which ties the global public IP address and port to the target proxy.
gcloud compute forwarding-rules create http-content-rule \
   --address=lb-ipv4-1\
   --global \
   --target-http-proxy=http-lb-proxy \
   --ports=80