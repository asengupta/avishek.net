<?php
	if (!class_exists('bbCodeTagSet', false)) {
		class bbCodeTagSet extends TagSet {
		
			function getTagSetName() {
				return 'bbCode';
			}
		
			function getExternalPluginInfos() {
				return array(
					'name' => 'BBCode',
					'website' => 'http://wordpress.org/extend/plugins/bbcode/'
				);
			}
		
			function isExternalPluginActive() {
				return class_exists('BBCode');
			}
		
			function isExternalPluginInstalled() {
				return is_dir($this->getPluginDir().'bbcode');
			}
		
		}
	}
	$tagset = new bbCodeTagSet();
?>