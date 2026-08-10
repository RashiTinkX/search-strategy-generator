"v=MCPv1; k=ed25519; p=Iwa2Vut33USQIVfV2dua5KEXKUs+c4MxAfqnRUX3dtU="


# Get your hosted zone ID
aws route53 list-hosted-zones-by-name --dns-name brainkb.org --query "HostedZones[0].Id" --output text

aws route53 change-resource-record-sets \
  --hosted-zone-id /hostedzone/Z06918342ADZVPCW09HXW \
  --change-batch '{
    "Changes": [{
      "Action": "UPSERT",
      "ResourceRecordSet": {
        "Name": "brainkb.org",
        "Type": "TXT",
        "TTL": 300,
        "ResourceRecords": [
          {"Value": "\"v=MCPv1; k=ed25519; p=Iwa2Vut33USQIVfV2dua5KEXKUs+c4MxAfqnRUX3dtU=\""}
        ]
      }
    }]
  }'