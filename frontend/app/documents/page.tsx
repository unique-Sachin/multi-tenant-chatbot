'use client';

import { useState, useEffect } from 'react';
import { ProtectedRoute } from '@/components/auth/protected-route';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Progress } from '@/components/ui/progress';
import { Badge } from '@/components/ui/badge';
import { Alert, AlertDescription } from '@/components/ui/alert';
import {
  Upload,
  FileText,
  CheckCircle2,
  XCircle,
  Loader2,
  RefreshCw,
  Download,
  Trash2,
} from 'lucide-react';
import { organizationService } from '@/lib/services/organizations';
import { apiClient } from '@/lib/api-client';
import type { Organization, Website, Document } from '@/lib/types';
import { toast } from 'sonner';
import Link from 'next/link';

export default function DocumentsPage() {
  const [organizations, setOrganizations] = useState<Organization[]>([]);
  const [websites, setWebsites] = useState<Website[]>([]);
  const [documents, setDocuments] = useState<Document[]>([]);
  const [selectedOrgId, setSelectedOrgId] = useState('');
  const [selectedNamespace, setSelectedNamespace] = useState('');
  const [isUploading, setIsUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [isLoading, setIsLoading] = useState(false);

  useEffect(() => {
    loadData();
  }, []);

  const loadData = async () => {
    setIsLoading(true);
    try {
      const [orgsData, websitesData] = await Promise.all([
        organizationService.getOrganizations(),
        organizationService.getWebsites(),
      ]);
      setOrganizations(orgsData);
      setWebsites(websitesData);
      
      // Load documents if implemented in backend
      // const docsData = await documentService.getDocuments();
      // setDocuments(docsData);
    } catch (error) {
      toast.error('Failed to load data');
    } finally {
      setIsLoading(false);
    }
  };

  const handleFileUpload = async (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    
    if (!selectedOrgId || !selectedNamespace) {
      toast.error('Please select an organization');
      return;
    }

    const formData = new FormData(e.currentTarget);
    const file = formData.get('file') as File;

    if (!file || file.size === 0) {
      toast.error('Please select a file');
      return;
    }

    // Validate file type
    const allowedTypes = ['application/pdf', 'text/plain', 'application/vnd.openxmlformats-officedocument.wordprocessingml.document'];
    if (!allowedTypes.includes(file.type)) {
      toast.error('Only PDF, TXT, and DOCX files are allowed');
      return;
    }

    // Validate file size (max 10MB)
    const maxSize = 10 * 1024 * 1024;
    if (file.size > maxSize) {
      toast.error('File size must be less than 10MB');
      return;
    }

    setIsUploading(true);
    setUploadProgress(0);

    try {
      // Simulate upload progress
      const progressInterval = setInterval(() => {
        setUploadProgress(prev => {
          if (prev >= 90) {
            clearInterval(progressInterval);
            return 90;
          }
          return prev + 10;
        });
      }, 200);

      await apiClient.uploadFile(
        '/documents/upload',
        file,
        {
          org_id: selectedOrgId,
          namespace: selectedNamespace,
        }
      );

      clearInterval(progressInterval);
      setUploadProgress(100);

      toast.success('Document uploaded successfully');
      (e.target as HTMLFormElement).reset();
      await loadData();
    } catch (error) {
      toast.error('Failed to upload document');
    } finally {
      setIsUploading(false);
      setTimeout(() => setUploadProgress(0), 1000);
    }
  };

  const handleOrganizationChange = (orgId: string) => {
    setSelectedOrgId(orgId);
    const website = websites.find(w => w.org_id === orgId);
    setSelectedNamespace(website?.namespace || '');
  };

  const getFileIcon = (fileType: string) => {
    return <FileText className="h-5 w-5" />;
  };

  const getStatusBadge = (status: string) => {
    switch (status) {
      case 'completed':
        return <Badge className="gap-1"><CheckCircle2 className="h-3 w-3" />Completed</Badge>;
      case 'failed':
        return <Badge variant="destructive" className="gap-1"><XCircle className="h-3 w-3" />Failed</Badge>;
      case 'processing':
        return <Badge variant="secondary" className="gap-1"><Loader2 className="h-3 w-3 animate-spin" />Processing</Badge>;
      default:
        return <Badge variant="outline">{status}</Badge>;
    }
  };

  const selectedOrg = organizations.find(o => o.id === selectedOrgId);

  return (
    <ProtectedRoute>
      <div className="min-h-screen bg-background">
        {/* Header */}
        <div className="border-b">
          <div className="container mx-auto px-4 py-4">
            <div className="flex items-center justify-between">
              <div>
                <h1 className="text-2xl font-bold">Document Upload</h1>
                <p className="text-muted-foreground">
                  Upload and manage documents for your organizations
                </p>
              </div>
              <div className="flex gap-2">
                <Link href="/chat">
                  <Button variant="outline">Back to Chat</Button>
                </Link>
                <Button onClick={loadData} disabled={isLoading}>
                  <RefreshCw className={`mr-2 h-4 w-4 ${isLoading ? 'animate-spin' : ''}`} />
                  Refresh
                </Button>
              </div>
            </div>
          </div>
        </div>

        {/* Main Content */}
        <div className="container mx-auto px-4 py-8 max-w-4xl space-y-6">
          {/* Upload Card */}
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Upload className="h-5 w-5" />
                Upload Document
              </CardTitle>
              <CardDescription>
                Upload PDF, TXT, or DOCX files (max 10MB)
              </CardDescription>
            </CardHeader>
            <CardContent>
              <form onSubmit={handleFileUpload} className="space-y-4">
                <div className="space-y-2">
                  <Label htmlFor="organization">Organization</Label>
                  <Select 
                    value={selectedOrgId} 
                    onValueChange={handleOrganizationChange}
                    required
                  >
                    <SelectTrigger>
                      <SelectValue placeholder="Select organization" />
                    </SelectTrigger>
                    <SelectContent>
                      {organizations.map((org) => (
                        <SelectItem key={org.id} value={org.id}>
                          {org.name}
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                  {selectedNamespace && (
                    <p className="text-xs text-muted-foreground">
                      Namespace: <span className="font-mono">{selectedNamespace}</span>
                    </p>
                  )}
                </div>

                <div className="space-y-2">
                  <Label htmlFor="file">File</Label>
                  <Input
                    id="file"
                    name="file"
                    type="file"
                    accept=".pdf,.txt,.docx"
                    required
                    disabled={!selectedOrgId}
                  />
                  <p className="text-xs text-muted-foreground">
                    Supported formats: PDF, TXT, DOCX (max 10MB)
                  </p>
                </div>

                {isUploading && uploadProgress > 0 && (
                  <div className="space-y-2">
                    <div className="flex justify-between text-sm">
                      <span className="text-muted-foreground">Uploading...</span>
                      <span className="font-medium">{uploadProgress}%</span>
                    </div>
                    <Progress value={uploadProgress} />
                  </div>
                )}

                <Button 
                  type="submit" 
                  disabled={!selectedOrgId || isUploading}
                  className="w-full"
                >
                  {isUploading ? (
                    <>
                      <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                      Uploading...
                    </>
                  ) : (
                    <>
                      <Upload className="mr-2 h-4 w-4" />
                      Upload Document
                    </>
                  )}
                </Button>
              </form>
            </CardContent>
          </Card>

          {/* Info Alert */}
          <Alert>
            <AlertDescription>
              <strong>Note:</strong> Documents will be processed and added to the organization's knowledge base.
              Processing may take a few minutes depending on document size.
            </AlertDescription>
          </Alert>

          {/* Documents List */}
          <Card>
            <CardHeader>
              <CardTitle>Uploaded Documents</CardTitle>
              <CardDescription>
                View and manage uploaded documents
              </CardDescription>
            </CardHeader>
            <CardContent>
              {documents.length === 0 ? (
                <p className="text-center text-muted-foreground py-8">
                  No documents uploaded yet
                </p>
              ) : (
                <div className="space-y-3">
                  {documents.map((doc) => {
                    const org = organizations.find(o => o.id === doc.org_id);
                    return (
                      <div
                        key={doc.id}
                        className="flex items-center justify-between p-4 border rounded-lg hover:bg-accent/50 transition-colors"
                      >
                        <div className="flex items-center gap-3 flex-1 min-w-0">
                          {getFileIcon(doc.file_type)}
                          <div className="flex-1 min-w-0">
                            <p className="font-medium truncate">{doc.filename}</p>
                            <p className="text-xs text-muted-foreground">
                              {org?.name} • {(doc.file_size / 1024).toFixed(2)} KB
                              {doc.chunks_created && ` • ${doc.chunks_created} chunks`}
                            </p>
                          </div>
                        </div>
                        <div className="flex items-center gap-2">
                          {getStatusBadge(doc.upload_status)}
                          <Button variant="ghost" size="icon">
                            <Trash2 className="h-4 w-4 text-destructive" />
                          </Button>
                        </div>
                      </div>
                    );
                  })}
                </div>
              )}
            </CardContent>
          </Card>
        </div>
      </div>
    </ProtectedRoute>
  );
}
