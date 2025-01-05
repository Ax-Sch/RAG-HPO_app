FROM ubuntu:22.04

# Install SSH server
RUN apt-get update && apt-get install -y openssh-server && \
    mkdir /var/run/sshd && \
    sed -i 's/#PermitRootLogin prohibit-password/PermitRootLogin no/' /etc/ssh/sshd_config

# Create an initialization script for user setup
COPY ssh_server/init.sh /init.sh
COPY ssh_server/users/users.txt /users.txt

RUN chmod +x /init.sh
RUN /init.sh
RUN rm /users.txt

# Expose SSH port
EXPOSE 22

RUN service ssh stop

